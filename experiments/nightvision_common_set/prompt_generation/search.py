"""Search for prompts with low-probability next tokens.

Flow:
  1. Seed iter 1: generate a fresh batch across character groups,
     evaluate, track the lowest-logprob prompt as the running seed.
  2. Seed iters 2..seed_iterations: generate `batch_size` candidates
     where half are 1-character permutations of the current best seed
     (greedy hill climbing) and half are fresh draws (exploration).
     Re-pick the seed greedily after each batch. Exit early once any
     prompt's probability falls below `seed_early_exit_probability`.
  3. Permutation iters: generate `batch_size` 1-character permutations
     of the current seed (split equally across enabled character groups),
     evaluate, then re-pick the seed greedily as the lowest-logprob
     prompt seen so far.
  4. End: sort every evaluated prompt by logprob ascending and save the
     bottom `target_count`.
"""

import json
import logging
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from api.chat_completions_client import ChatCompletionsClient, ChatResult

logger = logging.getLogger(__name__)

# -- Character group lists (ordered by Unicode block start) --
LATIN_CHARS = (  # U+0000–U+0020, U+0080–U+00FF, U+0180–U+024F  (369 chars)
    [chr(c) for c in range(0x0000, 0x0021)]
    + [chr(c) for c in range(0x0080, 0x0100)]
    + [chr(c) for c in range(0x0180, 0x0250)]
)
GREEK_AND_COPTIC = [chr(c) for c in range(0x0370, 0x0400)]                     # U+0370–U+03FF  (144 chars)
HEBREW_CHARS = [chr(c) for c in range(0x05D0, 0x05EB)]                         # U+05D0–U+05EA  (27 chars)
ARABIC_CHARS = [chr(c) for c in range(0x0620, 0x064B)]                         # U+0620–U+064A  (43 chars)
DEVANAGARI_CHARS = [chr(c) for c in range(0x0904, 0x093A)]                     # U+0904–U+0939  (54 chars)
THAI_CHARS = [chr(c) for c in range(0x0E01, 0x0E3B)]                           # U+0E01–U+0E3A  (58 chars)
COMBINING_DIACRITICAL = [chr(c) for c in range(0x0300, 0x0370)]                # U+0300–U+036F  (112 chars)
COMBINING_DIACRITICAL_EXT = [chr(c) for c in range(0x1AB0, 0x1AC0)]            # U+1AB0–U+1ABF  (16 chars)
COMBINING_DIACRITICAL_SUPPLEMENT = [chr(c) for c in range(0x1DC0, 0x1E00)]     # U+1DC0–U+1DFF  (64 chars)
COMBINING_DIACRITICAL_FOR_SYMBOLS = [chr(c) for c in range(0x20D0, 0x2100)]    # U+20D0–U+20FF  (48 chars)
INVISIBLE_ZERO_WIDTH = [                                                       # (25 chars)
    chr(0x00AD),        # Soft hyphen
    chr(0x034F),        # Combining grapheme joiner
    chr(0x061C),        # Arabic letter mark
    chr(0x200B),        # Zero-width space
    chr(0x200C),        # Zero-width non-joiner
    chr(0x200D),        # Zero-width joiner
    chr(0x200E),        # Left-to-right mark
    chr(0x200F),        # Right-to-left mark
    chr(0x202A),        # Left-to-right embedding
    chr(0x202B),        # Right-to-left embedding
    chr(0x202C),        # Pop directional formatting
    chr(0x202D),        # Left-to-right override
    chr(0x202E),        # Right-to-left override
    chr(0x2060),        # Word joiner
    chr(0x2061),        # Function application
    chr(0x2062),        # Invisible times
    chr(0x2063),        # Invisible separator
    chr(0x2064),        # Invisible plus
    chr(0x2066),        # Left-to-right isolate
    chr(0x2067),        # Right-to-left isolate
    chr(0x2068),        # First strong isolate
    chr(0x2069),        # Pop directional isolate
    chr(0xFEFF),        # Zero-width no-break space (BOM)
    chr(0xFFA0),        # Halfwidth Hangul filler
    chr(0x115F),        # Hangul choseong filler
]
GENERAL_PUNCTUATION = [chr(c) for c in range(0x2000, 0x2070)]                  # U+2000–U+206F  (112 chars)
SUPERSCRIPTS_AND_SUBSCRIPTS = [chr(c) for c in range(0x2070, 0x20A0)]          # U+2070–U+209F  (48 chars)
MATHEMATICAL_OPERATORS = [chr(c) for c in range(0x2200, 0x2300)]               # U+2200–U+22FF  (256 chars)
ENCLOSED_ALPHANUM = [chr(c) for c in range(0x2460, 0x24FF)]                    # U+2460–U+24FE  (159 chars)
BOX_DRAWING = [chr(c) for c in range(0x2500, 0x2580)]                          # U+2500–U+257F  (128 chars)
MISC_SYMBOLS = [chr(c) for c in range(0x2600, 0x2700)]                         # U+2600–U+26FF  (256 chars)
DINGBATS = [chr(c) for c in range(0x2700, 0x27C0)]                             # U+2700–U+27BF  (192 chars)
BRAILLE_CHARS = [chr(c) for c in range(0x2800, 0x2900)]                        # U+2800–U+28FF  (256 chars)
SUPPLEMENTAL_MATH_OPERATORS = [chr(c) for c in range(0x2A00, 0x2B00)]          # U+2A00–U+2AFF  (256 chars)
CJK_COMPATIBILITY = [chr(c) for c in range(0x3300, 0x3400)]                    # U+3300–U+33FF  (256 chars)
CJK_CHARS = [chr(c) for c in range(0x4E00, 0x4E00 + 100)]                     # U+4E00–U+4E63  (100 chars)
VARIATION_SELECTORS = [chr(c) for c in range(0xFE00, 0xFE10)]                  # U+FE00–U+FE0F  (16 chars)
COMBINING_HALF_MARKS = [chr(c) for c in range(0xFE20, 0xFE30)]                 # U+FE20–U+FE2F  (16 chars)
HALFWIDTH_FULLWIDTH = [chr(c) for c in range(0xFF00, 0xFFF0)]                  # U+FF00–U+FFEF  (240 chars)
SPECIALS = [chr(c) for c in range(0xFFF0, 0x10000)]                            # U+FFF0–U+FFFF  (16 chars)
IDEOGRAPHIC_SYMBOLS = [chr(c) for c in range(0x16FE0, 0x17000)]                # U+16FE0–U+16FFF  (32 chars)
MUSICAL_SYMBOLS = [chr(c) for c in range(0x1D100, 0x1D12C)]                    # U+1D100–U+1D12B  (44 chars)
MATH_ALPHANUMERIC = [chr(c) for c in range(0x1D400, 0x1D800)]                  # U+1D400–U+1D7FF  (1024 chars)
MAHJONG_TILES = [chr(c) for c in range(0x1F000, 0x1F030)]                      # U+1F000–U+1F02F  (48 chars)
DOMINO_TILES = [chr(c) for c in range(0x1F030, 0x1F0A0)]                       # U+1F030–U+1F09F  (112 chars)
PLAYING_CARDS = [chr(c) for c in range(0x1F0A0, 0x1F100)]                      # U+1F0A0–U+1F0FF  (96 chars)
MISC_SYMBOLS_PICTOGRAPHS = [chr(c) for c in range(0x1F300, 0x1F600)]           # U+1F300–U+1F5FF  (768 chars)
EMOTICONS = [chr(c) for c in range(0x1F600, 0x1F650)]                          # U+1F600–U+1F64F  (80 chars)
TRANSPORT_MAP_SYMBOLS = [chr(c) for c in range(0x1F680, 0x1F700)]              # U+1F680–U+1F6FF  (128 chars)
ALCHEMICAL_SYMBOLS = [chr(c) for c in range(0x1F700, 0x1F780)]                 # U+1F700–U+1F77F  (128 chars)
GEOMETRIC_SHAPES_EXT = [chr(c) for c in range(0x1F780, 0x1F800)]               # U+1F780–U+1F7FF  (128 chars)
SUPPLEMENTAL_SYMBOLS_PICTOGRAPHS = [chr(c) for c in range(0x1F900, 0x1FA00)]   # U+1F900–U+1F9FF  (256 chars)
SYMBOLS_EXTENDED_A = [chr(c) for c in range(0x1FA00, 0x1FA70)]                 # U+1FA00–U+1FA6F  (112 chars)
SYMBOLS_PICTOGRAPHS_EXTENDED_A = [chr(c) for c in range(0x1FA70, 0x1FB00)]     # U+1FA70–U+1FAFF  (144 chars)
DEPRECATED_TAGS = [chr(c) for c in range(0xE0000, 0xE0080)]                    # U+E0000–U+E007F  (128 chars)
VARIATION_SELECTORS_SUPPLEMENT = [chr(c) for c in range(0xE0100, 0xE01F0)]     # U+E0100–U+E01EF  (240 chars)

# -- Homoglyph / confusable scripts (visually resemble Latin, tokenize differently) --
CYRILLIC_CHARS = [chr(c) for c in range(0x0400, 0x0500)]                       # U+0400–U+04FF  (256 chars)
ARMENIAN_CHARS = [chr(c) for c in range(0x0530, 0x0590)]                       # U+0530–U+058F  (96 chars)
GEORGIAN_CHARS = [chr(c) for c in range(0x10A0, 0x1100)]                       # U+10A0–U+10FF  (96 chars)
CHEROKEE_CHARS = [chr(c) for c in range(0x13A0, 0x13F6)]                       # U+13A0–U+13F5  (86 chars)

# -- Private Use Area (undefined semantics → degenerate embeddings) --
PRIVATE_USE_AREA = [chr(c) for c in range(0xE000, 0xE100)]                     # U+E000–U+E0FF  (256 chars, subset of PUA)

# -- IPA extensions & spacing modifiers (distorted-Latin phonetic chars) --
IPA_EXTENSIONS = [chr(c) for c in range(0x0250, 0x02B0)]                       # U+0250–U+02AF  (96 chars)
SPACING_MODIFIER_LETTERS = [chr(c) for c in range(0x02B0, 0x0300)]             # U+02B0–U+02FF  (80 chars)

# -- Ancient / historic scripts (near-zero training representation) --
RUNIC_CHARS = [chr(c) for c in range(0x16A0, 0x16F1)]                          # U+16A0–U+16F0  (81 chars)
OGHAM_CHARS = [chr(c) for c in range(0x1680, 0x169D)]                          # U+1680–U+169C  (29 chars)
OLD_ITALIC_CHARS = [chr(c) for c in range(0x10300, 0x10324)]                   # U+10300–U+10323  (36 chars)
EGYPTIAN_HIEROGLYPHS = [chr(c) for c in range(0x13000, 0x13100)]               # U+13000–U+130FF  (256 chars, subset)
CUNEIFORM = [chr(c) for c in range(0x12000, 0x12100)]                          # U+12000–U+120FF  (256 chars, subset)

# -- Control characters (C0/C1 — known to break alignment) --
C0_CONTROLS = [chr(c) for c in range(0x0001, 0x0020)]                          # U+0001–U+001F  (31 chars, excludes NUL)
C1_CONTROLS = [chr(c) for c in range(0x0080, 0x00A0)]                          # U+0080–U+009F  (32 chars)

# -- Letterlike symbols / number forms / currency (near-ASCII confusables) --
LETTERLIKE_SYMBOLS = [chr(c) for c in range(0x2100, 0x2150)]                   # U+2100–U+214F  (80 chars)
NUMBER_FORMS = [chr(c) for c in range(0x2150, 0x2190)]                         # U+2150–U+218F  (64 chars)
CURRENCY_SYMBOLS = [chr(c) for c in range(0x20A0, 0x20D0)]                     # U+20A0–U+20CF  (48 chars)

# -- Yijing hexagrams (structured, rare, 64 systematic symbols) --
YIJING_HEXAGRAMS = [chr(c) for c in range(0x4DC0, 0x4E00)]                     # U+4DC0–U+4DFF  (64 chars)

# -- Miscellaneous technical / OCR (sparse in training data) --
MISC_TECHNICAL = [chr(c) for c in range(0x2300, 0x2400)]                       # U+2300–U+23FF  (256 chars)
OCR_CHARS = [chr(c) for c in range(0x2440, 0x2460)]                            # U+2440–U+245F  (32 chars)

# -- Additional South/SE Asian scripts (under-represented, complex tokenization) --
MYANMAR_CHARS = [chr(c) for c in range(0x1000, 0x109A)]                        # U+1000–U+1099  (154 chars)
TIBETAN_CHARS = [chr(c) for c in range(0x0F00, 0x0FDB)]                        # U+0F00–U+0FDA  (219 chars)
LAO_CHARS = [chr(c) for c in range(0x0E80, 0x0EDE)]                            # U+0E80–U+0EDD  (94 chars)
KHMER_CHARS = [chr(c) for c in range(0x1780, 0x17FA)]                          # U+1780–U+17F9  (122 chars)

# -- Ethiopic (under-represented African script) --
ETHIOPIC_CHARS = [chr(c) for c in range(0x1200, 0x1380)]                       # U+1200–U+137F  (384 chars)

# -- Emoji modifiers (skin tones / components without base — unusual tokenizer state) --
EMOJI_MODIFIERS = [                                                            # (9 chars)
    chr(0x1F3FB), chr(0x1F3FC), chr(0x1F3FD), chr(0x1F3FE), chr(0x1F3FF),     # Skin tones
    chr(0x200D),  # ZWJ (for dangling ZWJ sequences)
    chr(0xFE0E),  # Text presentation selector
    chr(0xFE0F),  # Emoji presentation selector
    chr(0x20E3),  # Combining enclosing keycap
]

# -- Syriac / Thaana / N'Ko (additional under-represented scripts) --
SYRIAC_CHARS = [chr(c) for c in range(0x0700, 0x074B)]                         # U+0700–U+074A  (75 chars)
THAANA_CHARS = [chr(c) for c in range(0x0780, 0x07B2)]                         # U+0780–U+07B1  (50 chars)
NKO_CHARS = [chr(c) for c in range(0x07C0, 0x07FB)]                            # U+07C0–U+07FA  (59 chars)
EMOJIS = [
    # Faces & emotions
    '😀','🥹','😎','🤯','🥳','😴','🤠','🫠','🤫','🤖',
    # People & gestures
    '👋','🤝','🙏','🤘','👌','👍','👎','✌️','🫶','💪',
    # Hearts & symbols
    '❤️','🧡','💛','💚','💙','💜','🖤','🤍','💖','✨',
    # Animals
    '🐶','🐱','🦊','🐻','🐼','🦁','🐯','🐸','🐵','🦄',
    # Nature & weather
    '🌸','🌻','🍁','🌵','🌊','🔥','🌈','⚡','❄️','🌙',
    # Food & drink
    '🍕','🍔','🍣','🌮','🥐','🍜','🍫','🍩','🍓','☕',
    # Activities & sports
    '⚽','🏀','🎾','🏈','🏓','🏄','🚴','🎮','🎸','🎨',
    # Travel & places
    '✈️','🚗','🚆','🚀','🗺️','🗽','🗼','🏝️','🏔️','🏕️',
    # Objects & tech
    '📱','💻','⌚','📷','🎧','🔋','💡','🔑','🧲','🧰',
    # Celebrations
    '🎉','🎁','🎂','🎈','🪩','🥂','🎇','🧨','🕯️','🏆',
]

CHAR_GROUPS = {
    #"ascii": list(string.printable),

    # -- Combined groups --
    # 1. Math (1536 chars)
    "math": MATH_ALPHANUMERIC + MATHEMATICAL_OPERATORS + SUPPLEMENTAL_MATH_OPERATORS,
    # 2. Emoji-adjacent (1641 chars)
    "emoji_adjacent": (MISC_SYMBOLS_PICTOGRAPHS + MISC_SYMBOLS
                       + SUPPLEMENTAL_SYMBOLS_PICTOGRAPHS
                       + SYMBOLS_PICTOGRAPHS_EXTENDED_A
                       + EMOTICONS + TRANSPORT_MAP_SYMBOLS
                       + EMOJI_MODIFIERS),
    # 3. Games/Tiles (256 chars)
    "games_tiles": PLAYING_CARDS + MAHJONG_TILES + DOMINO_TILES,
    # 4. CJK (388 chars)
    "cjk": CJK_CHARS + CJK_COMPATIBILITY + IDEOGRAPHIC_SYMBOLS,
    # 5. Variation selectors & specials (272 chars)
    "variation_combining": VARIATION_SELECTORS + VARIATION_SELECTORS_SUPPLEMENT + SPECIALS,
    # 6. Decorative/Drawing (576 chars)
    "decorative": DINGBATS + BOX_DRAWING + GEOMETRIC_SHAPES_EXT + ALCHEMICAL_SYMBOLS,
    # 7. South/SE Asian scripts (112 chars)
    "south_se_asian": DEVANAGARI_CHARS + THAI_CHARS,
    # 8. Invisible/zero-width characters (25 chars)
    "invisible": INVISIBLE_ZERO_WIDTH,

    # -- Individual groups (char count) --
    "greek_and_coptic": GREEK_AND_COPTIC,                      # 144 chars
    "arabic": ARABIC_CHARS,                                     # 43 chars
    "hebrew": HEBREW_CHARS,                                     # 27 chars
    "braille": BRAILLE_CHARS,                                   # 256 chars
    "enclosed_alphanum": ENCLOSED_ALPHANUM,                     # 159 chars
    "musical_symbols": MUSICAL_SYMBOLS,                         # 44 chars
    "deprecated_tags": DEPRECATED_TAGS,                         # 128 chars
    "halfwidth_fullwidth": HALFWIDTH_FULLWIDTH,                 # 240 chars
    "combining_marks": (COMBINING_DIACRITICAL + COMBINING_DIACRITICAL_EXT        # 272 chars
                        + COMBINING_DIACRITICAL_SUPPLEMENT
                        + COMBINING_DIACRITICAL_FOR_SYMBOLS + COMBINING_HALF_MARKS),
    "latin": LATIN_CHARS,                                       # 369 chars
    "superscripts_and_subscripts": SUPERSCRIPTS_AND_SUBSCRIPTS, # 48 chars
    "general_punctuation": GENERAL_PUNCTUATION,                 # 112 chars

    # -- Homoglyph / confusable scripts (Latin look-alikes, rare tokens) --
    "homoglyphs": CYRILLIC_CHARS + ARMENIAN_CHARS + GEORGIAN_CHARS + CHEROKEE_CHARS,  # 534 chars
    "emojis": EMOJIS,                                                       # 100 chars

    # -- Private Use Area (undefined codepoints → degenerate embeddings) --
    "private_use": PRIVATE_USE_AREA,                                   # 256 chars

    # -- IPA / phonetic (distorted-Latin that tokenizes differently) --
    "phonetic": IPA_EXTENSIONS + SPACING_MODIFIER_LETTERS,             # 176 chars

    # -- Ancient scripts (near-zero training representation → max entropy) --
    "ancient_scripts": (RUNIC_CHARS + OGHAM_CHARS + OLD_ITALIC_CHARS   # 658 chars
                        + EGYPTIAN_HIEROGLYPHS + CUNEIFORM),

    # -- Control characters (C0+C1 — known to break alignment/tokenization) --
    "control_chars": C0_CONTROLS + C1_CONTROLS,                        # 63 chars

    # -- Near-ASCII symbols (letterlike, number forms, currency) --
    "near_ascii_symbols": LETTERLIKE_SYMBOLS + NUMBER_FORMS + CURRENCY_SYMBOLS,  # 192 chars

    # -- Yijing hexagrams (rare, structured, systematic) --
    "yijing": YIJING_HEXAGRAMS,                                        # 64 chars

    # -- Technical / OCR (sparse in training data) --
    "technical": MISC_TECHNICAL + OCR_CHARS,                            # 288 chars

    # -- Extended South/SE Asian (under-represented, complex tokenization) --
    "extended_asian": MYANMAR_CHARS + TIBETAN_CHARS + LAO_CHARS + KHMER_CHARS,  # 589 chars

    # -- Ethiopic (under-represented African script) --
    "ethiopic": ETHIOPIC_CHARS,                                         # 384 chars

    # -- RTL/Bidi attack sequences (script chars + directional overrides) --
    "bidi_attack": (ARABIC_CHARS + HEBREW_CHARS + SYRIAC_CHARS         # 236 chars
                    + THAANA_CHARS + NKO_CHARS + INVISIBLE_ZERO_WIDTH),

    "cjk_original": CJK_CHARS,                                  # 27 chars                                                    # 100 chars

    "arabic_variation_selectors": ARABIC_CHARS + VARIATION_SELECTORS, # 59 chars
    "hebrew_variation_selectors": HEBREW_CHARS + VARIATION_SELECTORS, # 43 chars
    "emoji_variation_selectors": EMOJIS + VARIATION_SELECTORS, # 116 chars

}

# ---------------------------------------------------------------------------
# Config & result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    model: str
    max_prompt_chars: int = 5
    variable_length: bool = False
    max_iterations: int = 50
    seed_iterations: int = 40
    seed_early_exit_probability: float = 0.02
    target_count: int = 10000
    batch_size: int = 20000
    output_file: str = "results.json"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    quiet: bool = False
    extra_body: Optional[dict] = None
    max_gen_tokens: int = 1
    logprob_token_index: int = 0
    enabled_groups: Optional[list[str]] = None


@dataclass
class SearchResult:
    prompt: str
    token: str
    logprob: float
    probability: float
    group: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt generation helpers
# ---------------------------------------------------------------------------

def _generate_from_merged_groups(length: int, group_names: list[str]) -> tuple[str, str]:
    """Draw each character from a uniformly chosen group, then shuffle."""
    chars: list[str] = []
    mixed: list[str] = []
    for _ in range(length):
        name = random.choice(group_names)
        chars.append(random.choice(CHAR_GROUPS[name]))
        mixed.append(name)
    random.shuffle(chars)
    return "".join(chars), "-".join(mixed)


class _PromptGenerator:
    """Uniformly generates random prompts across enabled character groups."""

    def __init__(self, max_chars: int = 5, variable_length: bool = False,
                 enabled_groups: Optional[list[str]] = None):
        self.max_chars = max_chars
        self.variable_length = variable_length
        if enabled_groups is not None:
            self.group_names = [n for n in CHAR_GROUPS if n in enabled_groups]
        else:
            self.group_names = list(CHAR_GROUPS.keys())
        # Include a synthetic "mixed" pseudo-group that pulls per-character
        # from a random concrete group.
        self.all_names = self.group_names + ["mixed"]

    def _pick_length(self) -> int:
        if self.variable_length:
            return random.randint(1, self.max_chars)
        return self.max_chars

    def generate(self, count: int) -> list[tuple[str, str]]:
        prompts: list[tuple[str, str]] = []
        if not self.all_names or not self.group_names:
            return prompts

        # Equal allocation across groups; remainder distributed to leading groups
        base, remainder = divmod(count, len(self.all_names))
        allocations = {
            name: base + (1 if i < remainder else 0)
            for i, name in enumerate(self.all_names)
        }

        for name, alloc in allocations.items():
            for _ in range(alloc):
                length = self._pick_length()
                if name == "mixed":
                    p, group = _generate_from_merged_groups(length, self.group_names)
                else:
                    group = name
                    p = "".join(random.choices(CHAR_GROUPS[name], k=length))
                if p and p.strip():
                    prompts.append((p, group))
        return prompts


# ---------------------------------------------------------------------------
# Search orchestrator
# ---------------------------------------------------------------------------

class LowLogprobSearch:
    """Orchestrates the greedy hill-climbing search."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.found_prompts: set[str] = set()
        self.results: list[SearchResult] = []
        # (prompt, token, logprob, group)
        self._all_evaluated: list[tuple[str, str, float, str]] = []
        # Trimmed-but-unevaluated (prompt, group) pairs; drained to top up
        # future under-filled batches before falling back to resampling.
        self._pending_buffer: list[tuple[str, str]] = []
        # Permutation seeds whose permutation space couldn't fill a batch
        # even after retries; excluded from future seed selection so we
        # don't snap back to them via the greedy improve check.
        self._exhausted_seeds: set[str] = set()

        self._generator = _PromptGenerator(
            max_chars=config.max_prompt_chars,
            variable_length=config.variable_length,
            enabled_groups=config.enabled_groups,
        )
        self._client = ChatCompletionsClient(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.model,
        )
        self._output_path = Path(config.output_file)

    # -- evaluation ----------------------------------------------------------

    def _evaluate(self, prompts: list[str]) -> list[ChatResult]:
        return self._client.evaluate_batch(
            batch_messages=prompts,
            max_tokens=self.config.max_gen_tokens,
            temperature=0.0,
            logprobs=True,
            show_progress=not self.config.quiet,
            extra_body=self.config.extra_body,
            logprob_token_index=self.config.logprob_token_index,
        )

    @staticmethod
    def _prompt_from(result: ChatResult) -> str:
        if result.messages:
            for msg in reversed(result.messages):
                if msg.get("role") == "user":
                    return msg.get("content", "")
        return ""

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        cfg = {
            "model": self.config.model,
            "max_prompt_chars": self.config.max_prompt_chars,
            "variable_length": self.config.variable_length,
            "iterations": self.config.max_iterations,
            "target": self.config.target_count,
        }
        summary = self._summary()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(json.dumps(
            {"config": cfg, "summary": summary,
             "results": [asdict(r) for r in self.results]}, indent=2,
        ))

    def _summary(self) -> dict:
        if not self.results:
            return {"count": 0}
        logprobs = sorted(r.logprob for r in self.results)
        n = len(logprobs)
        pcts = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        def _pct(p: float) -> float:
            return logprobs[max(0, int(n * p) - 1)]

        logprob_stats = {"lowest": logprobs[0], "highest": logprobs[-1]}
        prob_stats = {"lowest": math.exp(logprobs[0]),
                      "highest": math.exp(logprobs[-1])}
        for p in pcts:
            v = _pct(p)
            key = f"p{int(p * 100)}"
            logprob_stats[key] = v
            prob_stats[key] = math.exp(v)
        return {"count": n, "logprob": logprob_stats, "probability": prob_stats}

    # -- filtering -----------------------------------------------------------

    def _dedup(self, tagged_prompts: list[tuple[str, str]],
               batch_seen: set[str]) -> list[tuple[str, str]]:
        """Filter out prompts already evaluated (`found_prompts`) or already
        present in the current iteration (`batch_seen`).

        Does NOT add to `found_prompts` — that happens in `_run_iteration`
        after trimming, so trimmed-off prompts remain eligible for re-use
        via the pending buffer or future re-discovery.
        """
        unique = []
        for p, group in tagged_prompts:
            if p and p not in self.found_prompts and p not in batch_seen:
                unique.append((p, group))
                batch_seen.add(p)
        return unique

    # -- processing ----------------------------------------------------------

    def _process_results(self, results: list[ChatResult],
                         prompt_groups: dict[str, str]) -> None:
        for result in results:
            if not result.success or result.logprob is None:
                continue
            prompt = self._prompt_from(result)
            group = prompt_groups.get(prompt, "unknown")
            if "-" in group:
                group = "mixed"
            self._all_evaluated.append((prompt, result.token or "", result.logprob, group))

    # -- permutation helper --------------------------------------------------

    def _generate_permutations(self, base_prompt: str, count: int) -> list[tuple[str, str]]:
        """Generate 1-character permutations of base_prompt, split equally across groups.

        Each permutation replaces exactly one position in base_prompt with a
        character drawn from one of the (enabled) character groups.
        """
        if not base_prompt or count <= 0:
            return []

        group_names = self._generator.group_names
        if not group_names:
            return []

        base, remainder = divmod(count, len(group_names))
        allocations = {name: base + (1 if i < remainder else 0)
                       for i, name in enumerate(group_names)}

        candidates: list[tuple[str, str]] = []
        n = len(base_prompt)
        for name, alloc in allocations.items():
            group = CHAR_GROUPS[name]
            for _ in range(alloc):
                pos = random.randrange(n)
                new_char = random.choice(group)
                new_prompt = base_prompt[:pos] + new_char + base_prompt[pos + 1:]
                candidates.append((new_prompt, name))
        return candidates

    def _generate_seed_mix(self, base_prompt: str, count: int) -> list[tuple[str, str]]:
        """Half 1-char permutations of base_prompt, half fresh draws.

        Used during seed iterations to combine greedy hill-climbing on the
        current best prompt with continued exploration of fresh candidates.
        """
        if count <= 0:
            return []
        half = count // 2
        other = count - half
        perms = self._generate_permutations(base_prompt, half) if base_prompt else []
        fresh = self._generator.generate(other)
        # If permutations couldn't be produced (no seed yet), fall back to
        # fully fresh so the batch isn't half-empty.
        if not perms:
            fresh.extend(self._generator.generate(half))
        return perms + fresh

    # -- iteration -----------------------------------------------------------

    # If the post-dedup batch drops below this fraction of batch_size, we
    # resample more candidates from the same generator rather than evaluating
    # a half-empty batch. MAX_RESAMPLE_RETRIES caps the attempts so plateaus
    # (e.g. a fixed seed whose permutation space is nearly exhausted) still
    # make forward progress.
    _MIN_BATCH_FILL_RATIO = 0.5
    _MAX_RESAMPLE_RETRIES = 5

    def _resample_loop(self, gen_fn, unique: list, batch_seen: set,
                       batch_size: int, min_fill: int) -> int:
        """Run up to `_MAX_RESAMPLE_RETRIES` with `gen_fn` to reach `min_fill`.

        Mutates `unique` and `batch_seen` in place. Returns total prompts
        requested for accounting in caller's dedup statistics.
        """
        requested = 0
        retries = 0
        while len(unique) < min_fill and retries < self._MAX_RESAMPLE_RETRIES:
            retries += 1
            needed = batch_size - len(unique)
            request = int(math.ceil(needed * 1.5))
            logger.info("Post-dedup batch %d < %d (%.0f%% of %d); "
                        "resampling %d more (retry %d/%d)",
                        len(unique), min_fill,
                        self._MIN_BATCH_FILL_RATIO * 100, batch_size,
                        request, retries, self._MAX_RESAMPLE_RETRIES)
            more = gen_fn(request)
            if not more:
                break
            requested += request
            unique.extend(self._dedup(more, batch_seen))
        return requested

    def _pick_p5_fallback_seed(
        self, exclude: Optional[str] = None
    ) -> Optional[tuple[str, float]]:
        """Pick a random prompt from the bottom-5% logprob pool of evaluations.

        Used to swap in a fresh permutation seed when the current seed's
        permutation space is too small to fill a deduplicated batch. Excludes
        `exclude` and any previously-exhausted seeds so a genuinely new seed
        is chosen. Returns (prompt, logprob), or None if no candidate exists.
        """
        if not self._all_evaluated:
            return None
        excluded = set(self._exhausted_seeds)
        if exclude is not None:
            excluded.add(exclude)
        candidates = [e for e in self._all_evaluated if e[0] not in excluded]
        if not candidates:
            return None
        sorted_by_lp = sorted(candidates, key=lambda e: e[2])
        p5_count = max(1, int(len(sorted_by_lp) * 0.05))
        chosen = random.choice(sorted_by_lp[:p5_count])
        return chosen[0], chosen[2]

    def _min_non_exhausted(self) -> Optional[tuple[str, str, float, str]]:
        """Lowest-logprob evaluation that isn't a previously-exhausted seed."""
        candidates = [e for e in self._all_evaluated
                      if e[0] not in self._exhausted_seeds]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e[2])

    def _run_iteration(self, batch_size: int,
                       gen_fn: Optional[callable] = None,
                       permutation_seed: Optional[str] = None,
                       ) -> Optional[tuple[str, float]]:
        """Generate candidates via `gen_fn`, dedup + resample, evaluate, record.

        gen_fn(count) -> list[(prompt, group)]. If None, fresh-generate from
        `self._generator`. The first call requests 1.5x batch_size to cushion
        typical dedup losses. If the unique batch is below 50% of batch_size,
        first drain from `self._pending_buffer` (excess from prior iterations),
        then resample up to `_MAX_RESAMPLE_RETRIES` times. Excess uniques
        above batch_size are pushed to the buffer instead of discarded.
        Prompts are added to `self.found_prompts` only after trimming, so
        anything that lands in the buffer remains available for re-use.

        permutation_seed: when set, the iteration is treated as a permutation
        pass over this seed. If `_MAX_RESAMPLE_RETRIES` are exhausted and the
        batch is still under-filled, a fallback seed is drawn from the p5
        percentile (excluding the current seed and any previously exhausted
        seeds) and one more resample loop runs over permutations of the new
        seed. When this triggers, `permutation_seed` is added to the
        exhausted set and (new_seed, new_seed_lp) is returned so the caller
        can adopt it. Returns None when no seed swap occurred.
        """
        if gen_fn is None:
            gen_fn = self._generator.generate

        batch_seen: set[str] = set()
        total_requested = int(math.ceil(batch_size * 1.5))
        raw = gen_fn(total_requested)
        if not raw:
            logger.warning("Failed to generate any candidates")
            return None

        unique = self._dedup(raw, batch_seen)

        min_fill = int(batch_size * self._MIN_BATCH_FILL_RATIO)

        # Drain pending buffer first when below fill ratio: these are
        # already-generated prompts trimmed off prior batches.
        drained_from_buffer = 0
        if len(unique) < min_fill and self._pending_buffer:
            while len(unique) < batch_size and self._pending_buffer:
                p, g = self._pending_buffer.pop()
                if p and p not in self.found_prompts and p not in batch_seen:
                    unique.append((p, g))
                    batch_seen.add(p)
                    drained_from_buffer += 1

        total_requested += self._resample_loop(
            gen_fn, unique, batch_seen, batch_size, min_fill,
        )

        fallback_seed_info: Optional[tuple[str, float]] = None
        if len(unique) < min_fill and permutation_seed is not None:
            fallback = self._pick_p5_fallback_seed(exclude=permutation_seed)
            if fallback is not None:
                new_seed, new_lp = fallback
                preview = new_seed if len(new_seed) <= 50 else new_seed[:50] + "..."
                logger.info(
                    "Permutation space exhausted for current seed; "
                    "switching to p5 fallback seed "
                    "(logprob=%.6f, probability=%.8f, len=%d, prompt=%r)",
                    new_lp, math.exp(new_lp), len(new_seed), preview,
                )
                fallback_seed_info = fallback
                self._exhausted_seeds.add(permutation_seed)
                gen_fn = lambda n, s=new_seed: self._generate_permutations(s, n)
                total_requested += self._resample_loop(
                    gen_fn, unique, batch_seen, batch_size, min_fill,
                )

        if len(unique) < min_fill:
            logger.warning("Batch still below %d after resampling; "
                           "proceeding with %d prompts",
                           min_fill, len(unique))

        gen_fn_kept = len(unique) - drained_from_buffer
        dedup_removed = total_requested - gen_fn_kept
        cap_trimmed = max(0, len(unique) - batch_size)
        if len(unique) > batch_size:
            # Preserve excess for future under-filled batches instead of
            # discarding. They're not yet in `found_prompts`, so a future
            # iteration could also re-discover them via gen_fn.
            self._pending_buffer.extend(unique[batch_size:])
            unique = unique[:batch_size]
        if not unique:
            logger.warning("All candidates were duplicates")
            return fallback_seed_info

        # Commit: only now add to `found_prompts`, so anything we did not
        # commit to evaluating stays eligible for the buffer / re-discovery.
        self.found_prompts.update(p for p, _ in unique)

        prompt_groups: dict[str, str] = {p: g for p, g in unique}
        prompts_only = [p for p, _ in unique]

        logger.info(
            "Evaluating %d unique prompts "
            "(%d duplicates removed, %d drained from buffer, "
            "%d trimmed to buffer; buffer size=%d)",
            len(prompts_only), dedup_removed, drained_from_buffer,
            cap_trimmed, len(self._pending_buffer),
        )

        results = self._evaluate(prompts_only)
        failed = sum(1 for r in results if not r.success)
        if failed:
            logger.warning("%d evaluations failed", failed)

        logprobs = sorted(r.logprob for r in results if r.success and r.logprob is not None)
        if logprobs:
            n = len(logprobs)

            def _pct(p: float) -> float:
                return logprobs[max(0, int(n * p) - 1)]

            pcts = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            lp_vals = [_pct(p) for p in pcts]
            lp_low, lp_high = logprobs[0], logprobs[-1]

            #pct_fmt = ", ".join(f"p{int(p * 100)}=%.6f" for p in pcts)
            #logger.info(
            #    "Batch logprob stats (n=%d): lowest=%.6f, " + pct_fmt + ", highest=%.6f",
            #    n, lp_low, *lp_vals, lp_high,)
            prob_fmt = ", ".join(f"p{int(p * 100)}=%.8f" for p in pcts)
            logger.info(
                "Batch probability stats (n=%d): lowest=%.8f, " + prob_fmt + ", highest=%.8f",
                n, math.exp(lp_low), *(math.exp(v) for v in lp_vals), math.exp(lp_high),
            )

        self._process_results(results, prompt_groups)
        return fallback_seed_info

    # -- main loop -----------------------------------------------------------

    def run(self) -> list[SearchResult]:
        logger.info("Starting search: iterations=%d, batch_size=%d, target=%d",
                    self.config.max_iterations, self.config.batch_size,
                    self.config.target_count)

        # Initial seed search: up to `seed_iterations` fresh-generation rounds,
        # exiting early once any prompt's probability falls below
        # `seed_early_exit_probability`.
        seed_iterations = max(1, self.config.seed_iterations)
        early_exit_lp = math.log(self.config.seed_early_exit_probability)
        seed_prompt = None
        seed_lp = math.inf
        for s in range(seed_iterations):
            if seed_prompt is None:
                logger.info("Seed iteration %d/%d (fresh generation)",
                            s + 1, seed_iterations)
                self._run_iteration(self.config.batch_size)
            else:
                logger.info("Seed iteration %d/%d (50%% permutations of best seed, 50%% fresh)",
                            s + 1, seed_iterations)
                current_seed = seed_prompt
                self._run_iteration(
                    self.config.batch_size,
                    gen_fn=lambda n, b=current_seed: self._generate_seed_mix(b, n),
                )

            best = self._min_non_exhausted()
            if best is None:
                continue

            cand_prompt, _, cand_lp, _ = best
            if cand_lp < seed_lp:
                seed_prompt, seed_lp = cand_prompt, cand_lp

            if seed_prompt is not None and seed_lp < early_exit_lp:
                logger.info(
                    "Seed probability %.8f < %.8f threshold; "
                    "ending seed search after %d iteration(s)",
                    math.exp(seed_lp), self.config.seed_early_exit_probability,
                    s + 1,
                )
                break

        if seed_prompt is None:
            logger.warning("No scored candidates during seed search; aborting")
            return self.results

        preview = seed_prompt if len(seed_prompt) <= 50 else seed_prompt[:50] + "..."
        logger.info("Final seed: logprob=%.6f, probability=%.8f, len=%d, prompt=%r",
                    seed_lp, math.exp(seed_lp), len(seed_prompt), preview)

        # Greedy permutation phase — permute current seed, then re-pick seed.
        permutation_iterations = max(0, self.config.max_iterations - 1)
        for i in range(permutation_iterations):
            logger.info("Permutation iteration %d/%d (1-char permutation of seed)",
                        i + 1, permutation_iterations)
            # Bind the current seed so resampling inside _run_iteration uses it.
            current_seed = seed_prompt
            fallback = self._run_iteration(
                self.config.batch_size,
                gen_fn=lambda n, s=current_seed: self._generate_permutations(s, n),
                permutation_seed=current_seed,
            )
            if fallback is not None:
                fb_seed, fb_lp = fallback
                logger.info(
                    "Adopting p5 fallback as new permutation seed: "
                    "logprob %.6f -> %.6f, probability %.8f -> %.8f",
                    seed_lp, fb_lp, math.exp(seed_lp), math.exp(fb_lp),
                )
                seed_prompt, seed_lp = fb_seed, fb_lp

            best = self._min_non_exhausted()
            if best is None:
                continue
            new_prompt, _, new_lp, _ = best
            if new_lp < seed_lp:
                logger.info(
                    "Seed improved: logprob %.6f -> %.6f, probability %.8f -> %.8f",
                    seed_lp, new_lp, math.exp(seed_lp), math.exp(new_lp),
                )
                seed_prompt, seed_lp = new_prompt, new_lp

        # Final selection: bottom-K by logprob (ascending).
        pool = sorted(self._all_evaluated, key=lambda e: e[2])
        selected = pool[:self.config.target_count]
        self.results = [
            SearchResult(prompt=p, token=t, logprob=lp,
                         probability=math.exp(lp), group=g)
            for p, t, lp, g in selected
        ]
        self._save()
        logger.info("Saved %d prompts (pool=%d)", len(self.results), len(pool))
        return self.results


def run_search(config: SearchConfig) -> list[SearchResult]:
    """Convenience function to run the search."""
    return LowLogprobSearch(config).run()
