"""vLLM server lifecycle management."""

import fcntl
import json
import logging
import math
import os
import random
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import requests

_PORT_LOCK_DIR = Path("/tmp")
_PORT_ALLOC_LOCK = _PORT_LOCK_DIR / "vllm_port_alloc.lock"


def port_used(host: str = "localhost", port: int = 8000) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except socket.error:
            return False


def _port_reserved(port: int) -> bool:
    """Check if a port has been reserved by another VLLMServer instance."""
    return (_PORT_LOCK_DIR / f"vllm_port_{port}.reserved").exists()


def _reserve_port(port: int) -> None:
    """Create a reservation marker file for the given port."""
    (_PORT_LOCK_DIR / f"vllm_port_{port}.reserved").touch()


def _release_port(port: int) -> None:
    """Remove the reservation marker file for the given port."""
    try:
        (_PORT_LOCK_DIR / f"vllm_port_{port}.reserved").unlink(missing_ok=True)
    except OSError:
        pass


def find_port(min_port: int = 2000, max_port: int = 65535) -> int:
    """Find and atomically reserve an available port.

    Uses a node-wide exclusive file lock so concurrent processes on the same
    host cannot select the same port simultaneously.  The caller is responsible
    for releasing the reservation (VLLMServer does this automatically via
    _release_port in stop/stop_force).
    """
    _PORT_ALLOC_LOCK.touch(exist_ok=True)
    with open(_PORT_ALLOC_LOCK, "r") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        while True:
            port = random.randint(min_port, max_port)
            if not port_used(port=port) and not _port_reserved(port):
                _reserve_port(port)
                return port

logger = logging.getLogger(__name__)


def load_hf_config(model: str, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Return the model's raw config.json (a local directory or a Hub repo id)."""
    local = Path(model) / "config.json"
    if local.is_file():
        path = local
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, "config.json", cache_dir=cache_dir)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def context_overrides(hf_config: dict[str, Any], max_model_len: int) -> dict[str, Any]:
    """HF config overrides needed for vLLM to serve prompts up to max_model_len.

    Returns {} when the model's native context already covers max_model_len.
    Otherwise max_position_embeddings is raised, which sizes the RoPE cos/sin
    cache for every position served. YaRN-scaled models (e.g. Olmo-3) need
    more: vLLM sizes their cache as original_max_position_embeddings * factor
    and ignores max_position_embeddings, so factor is raised to cover
    max_model_len too. Outputs past the trained context are meaningless, but
    prefill compute (and so latency) does not depend on it.

    The RoPE override targets ``rope_parameters``, the key vLLM's model code
    reads. vLLM applies dict-valued --hf-overrides only after it has already
    copied the HF ``rope_scaling`` into ``rope_parameters``, so overriding
    ``rope_scaling`` itself has no effect (vLLM 0.17, transformers 4.57).
    For composite configs the overrides are nested under ``text_config``,
    which vLLM applies recursively.
    """
    nested = "text_config" in hf_config
    cfg = hf_config.get("text_config", hf_config)
    max_pos = cfg["max_position_embeddings"]
    if max_model_len <= max_pos:
        return {}
    overrides: dict[str, Any] = {"max_position_embeddings": max_model_len}
    rope = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    if rope.get("rope_type", rope.get("type")) == "yarn":
        orig = rope["original_max_position_embeddings"]
        factor = float(math.ceil(max_model_len / orig))
        if factor > float(rope["factor"]):
            rope_parameters = {**rope, "factor": factor}
            # vLLM patches rope_theta into rope_parameters and models read it
            # from there, so a wholesale replacement must carry it along.
            if "rope_theta" not in rope_parameters and "rope_theta" in cfg:
                rope_parameters["rope_theta"] = cfg["rope_theta"]
            overrides["rope_parameters"] = rope_parameters
    return {"text_config": overrides} if nested else overrides


class VLLMServer:
    """Manages vLLM server lifecycle with context manager support."""

    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = None,
        max_num_seqs: int = None,
        flash_attn_version: int = None,
        container_image: str | Path | None = None,
        hf_cache_dir: str | Path | None = None,
        log_dir: str | Path = "logs/vllm",
        extra_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        hf_overrides: dict[str, Any] | None = None,
        extend_context: bool = False,
    ):
        """Initialize vLLM server manager.

        Args:
            model: Model name/path to serve
            host: Server host address
            port: Server port
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum model context length
            max_num_seqs: Maximum number of sequences per batch
            flash_attn_version: Flash attention version (2 or 3). None for auto-detect.
            container_image: Path to Singularity .sif image. Defaults to $PROJECT/containers/vllm.sif.
            hf_cache_dir: HuggingFace cache directory for bind mount. Defaults to $HF_HOME or ~/.cache/huggingface.
            log_dir: Directory for server logs
            extra_args: Additional raw vLLM CLI arguments (e.g. ["--no-enable-prefix-caching"]).
                They are joined into a shell command, so quote values with shlex.quote.
            extra_env: Additional environment variables exported inside the container
            hf_overrides: Values to override in the served HF config (vLLM --hf-overrides).
                Only top-level keys are replaced.
            extend_context: If max_model_len exceeds the model's native context, add the
                HF config overrides that let vLLM serve it anyway (see context_overrides)
                and lift vLLM's max_model_len check. Needs the model's config.json.
        """
        self.model = model
        self.host = host
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.flash_attn_version = flash_attn_version
        self.extra_args = list(extra_args or [])
        self.extra_env = dict(extra_env or {})
        self.container_image = Path(
            container_image
            or os.path.join(os.environ.get("PROJECT", "."), "containers", "vllm.sif")
        )
        # Resolve the HuggingFace *hub* cache directory (the dir that holds the
        # ``models--*`` folders). This path is both bind-mounted into the
        # container and exported as HF_HUB_CACHE inside it, so vLLM loads the
        # staged weights instead of re-downloading. Prefer HF_HUB_CACHE since
        # that is the variable that actually controls where weights live; fall
        # back to $HF_HOME/hub, then the default cache.
        if hf_cache_dir is not None:
            resolved_cache = str(hf_cache_dir)
        elif os.environ.get("HF_HUB_CACHE"):
            resolved_cache = os.environ["HF_HUB_CACHE"]
        elif os.environ.get("HF_HOME"):
            resolved_cache = os.path.join(os.environ["HF_HOME"], "hub")
        else:
            resolved_cache = os.path.expanduser("~/.cache/huggingface/hub")
        self.hf_cache_dir = Path(resolved_cache)
        self.hf_overrides = dict(hf_overrides or {})
        if extend_context and max_model_len is not None:
            hf_config = load_hf_config(model, self.hf_cache_dir)
            overrides = context_overrides(hf_config, max_model_len)
            if overrides:
                logger.warning(f"Extending {model} context to {max_model_len}: {overrides}")
                self.hf_overrides.update(overrides)
                # vLLM derives its max_model_len limit from the config (for some
                # models from rope_scaling rather than max_position_embeddings).
                self.extra_env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        self.log_dir = Path(log_dir)
        self.process: subprocess.Popen[bytes] | None = None

        # Create safe filename from model name
        safe_model_name = model.replace("/", "_").replace("\\", "_")
        self.log_file_path = self.log_dir / f"{safe_model_name}_server.log"

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_url(self) -> str:
        """Get the base URL for the vLLM OpenAI-compatible API."""
        return f"http://{self.host}:{self.port}/v1"

    def start(self, timeout: int = 1600, max_bind_retries: int = 5) -> None:
        """Start vLLM server and wait until healthy.

        If the vLLM subprocess dies because the chosen port was grabbed by
        another process between reservation and bind (OSError errno 98), the
        server will release that port, pick a new one via ``find_port``, and
        relaunch, up to ``max_bind_retries`` times.

        Args:
            timeout: Maximum seconds to wait for server to become healthy
            max_bind_retries: Number of times to re-pick a port and relaunch
                when the process dies with "Address already in use"

        Raises:
            TimeoutError: If server doesn't become healthy within timeout
            RuntimeError: If server process fails to start
        """
        if self.process is not None:
            logger.warning("Server already running")
            return

        attempt = 0
        while True:
            # Kill any orphaned processes on the port before starting
            self._kill_orphaned_processes()

            logger.info(f"Starting vLLM server for {self.model} on port {self.port}")
            logger.info(f"Log file: {self.log_file_path}")

            # Build command
            cmd = self._build_vllm_command()
            logger.debug(f"Command: {' '.join(cmd)}")

            # Open log file and keep the handle so we can close it in stop()
            self._log_file_handle = open(self.log_file_path, "w")

            # Start subprocess in its own process group so we can kill all children
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=self._log_file_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as e:
                self._close_log_file()
                raise RuntimeError(f"Failed to start vLLM server: {e}")

            # Wait for server to become healthy
            try:
                self._wait_for_healthy(timeout)
                logger.info(f"vLLM server ready at http://{self.host}:{self.port}")
                return
            except RuntimeError:
                # Process died. If it died because another process grabbed the
                # port after we reserved it, pick a new port and retry.
                self.stop_force()
                if attempt < max_bind_retries and self._log_indicates_bind_error():
                    attempt += 1
                    old_port = self.port
                    self.port = find_port()
                    logger.warning(
                        f"Port {old_port} bind failed (Address already in use); "
                        f"retrying with port {self.port} "
                        f"(attempt {attempt}/{max_bind_retries})"
                    )
                    continue
                raise
            except Exception:
                self.stop_force()
                raise

    def _close_log_file(self) -> None:
        """Close the server log file handle if open."""
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except OSError:
                pass
            self._log_file_handle = None

    def stop(self, timeout: int = 30) -> None:
        """Stop vLLM server gracefully.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown
        """
        if self.process is None:
            return

        logger.info(f"Stopping vLLM server for {self.model}")

        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.process.wait(timeout=timeout)
            logger.info("Server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning(f"Server didn't stop within {timeout}s, force killing")
            self.stop_force()
        except (ProcessLookupError, OSError) as e:
            logger.warning(f"Process group not found during stop: {e}")
        finally:
            self.process = None
            self._close_log_file()
            self._cleanup_gpu_memory()
            _release_port(self.port)

    def stop_force(self, timeout: int = 60) -> None:
        """Force stop vLLM server and all child processes."""
        if self.process is None:
            return

        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.process.wait(timeout=timeout)
            logger.info("Server process group force killed")
        except (ProcessLookupError, OSError) as e:
            logger.warning(f"Process group not found during force kill: {e}")
            try:
                self.process.kill()
                self.process.wait(timeout=timeout)
            except Exception:
                pass
        finally:
            self.process = None
            self._close_log_file()
            self._cleanup_gpu_memory()
            _release_port(self.port)

    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/health", timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory after server shutdown."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU cache cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")

        time.sleep(2)

    def _log_indicates_bind_error(self) -> bool:
        """Scan the server log for an "Address already in use" bind failure."""
        try:
            with open(self.log_file_path, "r", errors="replace") as f:
                contents = f.read()
        except OSError:
            return False
        return "Errno 98" in contents or "Address already in use" in contents

    def _kill_orphaned_processes(self) -> None:
        """Kill any orphaned vLLM processes using the configured port."""
        try:
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{self.port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        pid_int = int(pid)
                        logger.warning(f"Killing orphaned process {pid_int} on port {self.port}")
                        os.kill(pid_int, signal.SIGKILL)
                    except (ValueError, ProcessLookupError):
                        pass
                time.sleep(2)
                self._cleanup_gpu_memory()
            # Clear any stale reservation left by a crashed previous run.
            _release_port(self.port)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to check for orphaned processes: {e}")

    def _build_vllm_command(self) -> list[str]:
        """Build vLLM command launched via Singularity container."""
        vllm_args = [
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            #"--load-format", "fastsafetensors",
        ]
        if self.max_num_seqs is not None:
            vllm_args += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.max_model_len is not None:
            vllm_args += ["--max-model-len", str(self.max_model_len)]
        if self.max_num_seqs is not None and self.max_model_len is not None:
            vllm_args += ["--max-num-batched-tokens", str(self.max_num_seqs * self.max_model_len)]
        if self.flash_attn_version is not None:
            vllm_args += ["--attention-config", "'" + json.dumps({"flash_attn_version": self.flash_attn_version}) + "'"]

        if self.hf_overrides:
            vllm_args += ["--hf-overrides", shlex.quote(json.dumps(self.hf_overrides))]
        vllm_args += ["--generation-config", "vllm"]
        vllm_args += self.extra_args

        inner_script = (
            "export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache && "
            "export TMPDIR=/tmp && "
            "export HF_HUB_DISABLE_XET=1 && "
            # Singularity mounts $HOME, so ignore ~/.local site-packages; a
            # user-installed torch there would shadow the image's and break vLLM.
            "export PYTHONNOUSERSITE=1 && "
            # Override the HF_HUB_CACHE baked into the .sif image so vLLM reads
            # the bind-mounted (possibly node-local) cache instead of the image
            # default. A host-side export alone is overridden by the image env.
            f"export HF_HUB_CACHE={self.hf_cache_dir} && "
            + "".join(f"export {k}={shlex.quote(v)} && " for k, v in self.extra_env.items())
            + "python3 -m vllm.entrypoints.openai.api_server " + " ".join(vllm_args)
        )

        cmd = [
            "singularity", "exec", "--nv", "--writable-tmpfs",
            "--bind", f"{Path.home() / '.cache' / 'vllm'}:/root/.cache/vllm",
            "--bind", "/tmp:/tmp",
            "--bind", f"{self.hf_cache_dir}:{self.hf_cache_dir}",
            str(self.container_image),
            "bash", "-c", inner_script,
        ]
        return cmd

    def _wait_for_healthy(self, timeout: int) -> None:
        """Poll health endpoint until server is ready."""
        start_time = time.time()
        last_log_time = start_time

        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process died with exit code {self.process.returncode}. "
                    f"Check logs at {self.log_file_path}"
                )

            if self.is_healthy():
                return

            if time.time() - last_log_time > 30:
                elapsed = int(time.time() - start_time)
                logger.info(f"Still waiting for server to be ready ({elapsed}s elapsed)...")
                last_log_time = time.time()

            time.sleep(5)

        raise TimeoutError(
            f"vLLM server failed to become healthy within {timeout}s. "
            f"Check logs at {self.log_file_path}"
        )

    def __enter__(self) -> "VLLMServer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.stop()