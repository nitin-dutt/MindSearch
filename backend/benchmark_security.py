"""AES-GCM micro-benchmark for the Encryptor used to protect chunks at rest.

Run from backend/:  python benchmark_security.py
Prints hardware info and p50/p95 latency per chunk size so README numbers are reproducible.
"""
import platform
import statistics
import time

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from encryptor import Encryptor

CHUNK_CHARS = [300, 600, 1200]
WARMUP = 200
ITERATIONS = 2000


def _percentile(sorted_ms, pct):
    return sorted_ms[min(len(sorted_ms) - 1, int(len(sorted_ms) * pct))]


def _time_ms(fn):
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def bench(chunk_chars):
    enc = Encryptor(key=AESGCM.generate_key(bit_length=256))  # throwaway key, never touches encryption.key
    text = ("This is a sample sentence to test AES-256 encryption latency. " * 40)[:chunk_chars]
    blob = enc.encrypt(text)

    for _ in range(WARMUP):
        enc.decrypt(enc.encrypt(text))

    enc_ms = sorted(_time_ms(lambda: enc.encrypt(text)) for _ in range(ITERATIONS))
    dec_ms = sorted(_time_ms(lambda: enc.decrypt(blob)) for _ in range(ITERATIONS))
    return enc_ms, dec_ms


def main():
    print("=== AES-GCM-256 chunk encryption benchmark ===")
    print(f"Python {platform.python_version()} | {platform.processor() or platform.machine()} | {platform.system()} {platform.release()}")
    print(f"{ITERATIONS} iterations after {WARMUP} warm-up\n")
    print(f"{'chars':>6} | {'op':<7} | {'p50 ms':>8} | {'p95 ms':>8} | {'chunks/s (p50)':>14}")
    for chars in CHUNK_CHARS:
        enc_ms, dec_ms = bench(chars)
        for name, vals in (("encrypt", enc_ms), ("decrypt", dec_ms)):
            p50 = statistics.median(vals)
            print(f"{chars:>6} | {name:<7} | {p50:>8.4f} | {_percentile(vals, 0.95):>8.4f} | {1000 / p50:>14,.0f}")


if __name__ == "__main__":
    main()
