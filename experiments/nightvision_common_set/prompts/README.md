# Prompt sets

This directory holds one prompt set per model stem, in a file named
`<stem>.json` with the form `{"prompts": [...]}`. A set contains the 80000
prompts found by `../prompt_generation/` with the lowest probability for the
model's most likely next token. The lowest-probability prompts come first.
Duplicates are removed both by string and by token-id sequence.

`../prompt_generation/postprocess.py` writes these files.
`../collect_logits/` reads them, using only the first 20000 prompts of each set
by default. The sampling stages see the prompts only through the logits
collected from them.

The `.json` files (about 5 to 10 MB each) are tracked in git. To regenerate
them from the raw search results, run `python ../prompt_generation/postprocess.py`.

`prompt_stats.txt` is tracked. For each set it gives the prompt count and the
percentiles of the most likely next token's probability across the set's
prompts. It contains the same table that `postprocess.py` writes to
`../prompt_generation/results/probability_summary.txt` (not tracked), without the
average-token column.

Prompt sets can also be placed in subdirectories, for example
`variant/<stem>.json`. In that case `collect_all.sh` writes their logits to a
matching subdirectory and appends the variant name to the output filename.
