import gradio as gr
import numpy as np
from PIL import Image
import hashlib
import math
import os
import tempfile

# =========================
# OPTIONAL DEPENDENCIES
# =========================
# Audio: soundfile (recommended) OR scipy fallback
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False
    from scipy.io import wavfile

# =========================
# CONFIG
# =========================
IMG_SIZE = 256
MARKER = b"STEGv2|"
HEADER_BITS = 32

# =========================
# IMAGE UTILS
# =========================
def ensure_u8_rgb(img) -> np.ndarray:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def resize_u8(img_u8: np.ndarray, size=IMG_SIZE) -> np.ndarray:
    pil = Image.fromarray(img_u8)
    pil = pil.resize((size, size), Image.BICUBIC)
    return np.array(pil, dtype=np.uint8)

def diff_map_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    diff = np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).mean(axis=2)
    mx = float(diff.max())
    if mx <= 1e-9:
        d = diff.astype(np.uint8)
    else:
        d = (diff / mx * 255.0).astype(np.uint8)
    return np.stack([d, d, d], axis=-1)

def psnr(a_u8, b_u8) -> float:
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))

# =========================
# CRYPTO / PACKING UTILS
# =========================
def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def xor_bytes(data: bytes, key: str) -> bytes:
    k = key.encode("utf-8")
    if len(k) == 0:
        return data
    return bytes([b ^ k[i % len(k)] for i, b in enumerate(data)])

def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)
    return np.packbits(bits).tobytes()

def u32_to_bits(x: int) -> np.ndarray:
    out = np.zeros((32,), dtype=np.uint8)
    for i in range(32):
        out[i] = (x >> (31 - i)) & 1
    return out

def bits_to_u32(bits32: np.ndarray) -> int:
    x = 0
    for i in range(32):
        x = (x << 1) | int(bits32[i] & 1)
    return x

# =========================
# FAST HEATMAP (NO OPENCV)
# =========================
def compute_heatmap(img_u8: np.ndarray) -> np.ndarray:
    img = img_u8.astype(np.float32)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    g = np.pad(gray, ((1, 1), (1, 1)), mode="edge")

    gx = (
        (1.0 * g[:-2, :-2] + 0.0 * g[:-2, 1:-1] - 1.0 * g[:-2, 2:]) +
        (2.0 * g[1:-1, :-2] + 0.0 * g[1:-1, 1:-1] - 2.0 * g[1:-1, 2:]) +
        (1.0 * g[2:, :-2] + 0.0 * g[2:, 1:-1] - 1.0 * g[2:, 2:])
    )
    gy = (
        (1.0 * g[:-2, :-2] + 2.0 * g[:-2, 1:-1] + 1.0 * g[:-2, 2:]) +
        (0.0 * g[1:-1, :-2] + 0.0 * g[1:-1, 1:-1] + 0.0 * g[1:-1, 2:]) +
        (-1.0 * g[2:, :-2] - 2.0 * g[2:, 1:-1] - 1.0 * g[2:, 2:])
    )

    mag = np.sqrt(gx * gx + gy * gy)
    mx = float(mag.max())
    if mx <= 1e-9:
        heat = np.ones_like(mag, dtype=np.float32) * 0.5
    else:
        heat = (mag / mx).astype(np.float32)

    heat = (heat + np.roll(heat, 1, 0) + np.roll(heat, -1, 0) +
            np.roll(heat, 1, 1) + np.roll(heat, -1, 1)) / 5.0
    return np.clip(heat, 0, 1)

def heat_overlay(img_u8: np.ndarray, heat: np.ndarray, strength=0.62) -> np.ndarray:
    h = np.clip(heat, 0, 1)
    r = (np.clip(2 * h, 0, 1) * 255).astype(np.uint8)
    g = (np.clip(2 * h - 0.35, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1 - 2 * h, 0, 1) * 255).astype(np.uint8)
    cmap = np.stack([r, g, b], axis=-1)
    out = img_u8.astype(np.float32) * (1 - strength) + cmap.astype(np.float32) * strength
    return np.clip(out, 0, 255).astype(np.uint8)

# =========================
# ECC (Repetition + Majority vote)
# =========================
def ecc_repeat(bits: np.ndarray, r: int) -> np.ndarray:
    if r <= 1:
        return bits.astype(np.uint8)
    return np.repeat(bits.astype(np.uint8), r)

def ecc_majority(bits_rep: np.ndarray, r: int) -> np.ndarray:
    if r <= 1:
        return bits_rep.astype(np.uint8)
    n = len(bits_rep) // r
    bits_rep = bits_rep[:n * r].reshape(n, r)
    return (bits_rep.sum(axis=1) >= (r / 2)).astype(np.uint8)

# =========================
# POSITIONING
# =========================
def spread_select(order: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if n >= len(order):
        return order.copy()
    idx = np.linspace(0, len(order) - 1, n).astype(np.int64)
    return order[idx]

# =========================
# IMAGE STEGO CORE
# =========================
def embed_bits(cover_u8: np.ndarray, bits_header_rep: np.ndarray, bits_payload_rep: np.ndarray, heat: np.ndarray, bpc: int):
    h, w, _ = cover_u8.shape
    stego = cover_u8.copy()

    order = np.argsort(heat.flatten())[::-1].astype(np.int64)
    bits_per_pixel = 3 * bpc

    header_slots = int(math.ceil(len(bits_header_rep) / bits_per_pixel))
    if header_slots >= len(order):
        return None, None, "Image too small for header."

    header_pixels = order[:header_slots]
    remaining = order[header_slots:]

    payload_slots = int(math.ceil(len(bits_payload_rep) / bits_per_pixel))
    if payload_slots > len(remaining):
        return None, None, "Payload too large (capacity exceeded)."

    payload_pixels = spread_select(remaining, payload_slots)

    def embed_into(pixels, bits):
        idx = 0
        for c in pixels:
            y = int(c // w); x = int(c % w)
            for ch in range(3):
                v = int(stego[y, x, ch])
                for bitpos in range(bpc):
                    if idx >= len(bits):
                        break
                    mask = (1 << bitpos)
                    v = (v & (255 ^ mask)) | (int(bits[idx]) << bitpos)
                    idx += 1
                stego[y, x, ch] = np.uint8(v)
                if idx >= len(bits):
                    break
            if idx >= len(bits):
                break
        return idx

    if embed_into(header_pixels, bits_header_rep) < len(bits_header_rep):
        return None, None, "Failed to embed full header."
    if embed_into(payload_pixels, bits_payload_rep) < len(bits_payload_rep):
        return None, None, "Failed to embed full payload."

    return stego, (header_slots, payload_slots), "OK"

def extract_bits(img_u8: np.ndarray, heat: np.ndarray, bpc: int, r: int):
    h, w, _ = img_u8.shape
    order = np.argsort(heat.flatten())[::-1].astype(np.int64)
    bits_per_pixel = 3 * bpc

    header_rep_len = HEADER_BITS * r
    header_slots = int(math.ceil(header_rep_len / bits_per_pixel))
    if header_slots >= len(order):
        return None, "Cannot read header."

    header_pixels = order[:header_slots]
    remaining = order[header_slots:]

    header_rep = []
    for c in header_pixels:
        y = int(c // w); x = int(c % w)
        for ch in range(3):
            v = int(img_u8[y, x, ch])
            for bitpos in range(bpc):
                header_rep.append((v >> bitpos) & 1)
                if len(header_rep) >= header_rep_len:
                    break
            if len(header_rep) >= header_rep_len:
                break
        if len(header_rep) >= header_rep_len:
            break

    header_rep = np.array(header_rep[:header_rep_len], dtype=np.uint8)
    header_bits = ecc_majority(header_rep, r)
    if len(header_bits) < 32:
        return None, "Header decode failed."

    L = bits_to_u32(header_bits[:32])
    if L <= 0:
        return None, "Invalid payload length."

    payload_rep_len = L * r
    payload_slots = int(math.ceil(payload_rep_len / bits_per_pixel))
    if payload_slots > len(remaining):
        return None, "Payload exceeds capacity (tampered?)."

    payload_pixels = spread_select(remaining, payload_slots)

    payload_rep = []
    for c in payload_pixels:
        y = int(c // w); x = int(c % w)
        for ch in range(3):
            v = int(img_u8[y, x, ch])
            for bitpos in range(bpc):
                payload_rep.append((v >> bitpos) & 1)
                if len(payload_rep) >= payload_rep_len:
                    break
            if len(payload_rep) >= payload_rep_len:
                break
        if len(payload_rep) >= payload_rep_len:
            break

    payload_rep = np.array(payload_rep[:payload_rep_len], dtype=np.uint8)
    payload_bits = ecc_majority(payload_rep, r)[:L]

    total = np.concatenate([header_bits[:32], payload_bits], axis=0)
    return total, "OK"

# =========================
# PACK / UNPACK MESSAGE
# =========================
def pack_message_bits(message: str, key: str):
    msg_b = message.encode("utf-8")
    payload = MARKER + msg_b + b"|HASH|" + sha256_hex_bytes(msg_b).encode("utf-8")
    enc = xor_bytes(payload, key)
    data_bits = bytes_to_bits(enc)
    header_bits = u32_to_bits(len(data_bits))
    return header_bits, data_bits

def unpack_message_bits(total_bits: np.ndarray, key: str):
    if len(total_bits) < 32:
        return False, "Not enough bits."
    L = bits_to_u32(total_bits[:32])
    if L <= 0 or 32 + L > len(total_bits):
        return False, "Invalid length header."

    data_bits = total_bits[32:32 + L]
    data = bits_to_bytes(data_bits)
    dec = xor_bytes(data, key)

    if not dec.startswith(MARKER):
        return False, "Wrong key or tampered (marker missing)."

    body = dec[len(MARKER):]
    if b"|HASH|" not in body:
        return False, "Tampered (HASH missing)."

    msg_part, hash_part = body.split(b"|HASH|", 1)
    calc = sha256_hex_bytes(msg_part).encode("utf-8")
    if hash_part[:len(calc)] != calc:
        return False, "Integrity FAIL (hash mismatch)."

    return True, msg_part.decode("utf-8", errors="ignore")

# =========================
# IMAGE EVE ATTACKS
# =========================
def eve_attack(img_u8: np.ndarray, mode: str, strength: int, crop_ratio: float):
    if img_u8 is None:
        return None, "No image."
    out = img_u8.copy()
    h, w, _ = out.shape

    if mode == "None":
        return out, "Forwarded as-is."

    if mode == "Noise":
        ns = int(strength)
        noise = np.random.randint(-ns, ns + 1, size=out.shape, dtype=np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return out, f"Noise ±{ns}"

    if mode == "Crop + Resize":
        cr = float(crop_ratio)
        new_w = max(8, int(w * cr))
        new_h = max(8, int(h * cr))
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        pil = Image.fromarray(out).crop((x0, y0, x0 + new_w, y0 + new_h)).resize((w, h), Image.BICUBIC)
        return np.array(pil, dtype=np.uint8), f"Cropped to {int(cr * 100)}% then resized"

    if mode == "JPEG-like (Quantize)":
        q = max(2, int(strength))
        out = (out // q) * q
        return out.astype(np.uint8), f"Quantized step={q}"

    if mode == "Pixel Tamper":
        flips = max(200, int(strength) * 250)
        flat = out.reshape(-1)
        idx = np.random.randint(0, flat.size, size=flips)
        flat[idx] ^= 1
        return out, f"Flipped {flips} random LSBs"

    return out, "Unknown mode"

# =========================
# AUDIO UTILS + STEGO
# =========================
def read_wav(path):
    if HAVE_SF:
        audio, sr = sf.read(path, dtype="int16")
        return audio, int(sr)
    sr, audio = wavfile.read(path)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio, int(sr)

def write_wav(path, audio_int16, sr):
    if HAVE_SF:
        sf.write(path, audio_int16, sr, subtype="PCM_16")
    else:
        wavfile.write(path, sr, audio_int16)

def audio_capacity_bits(audio_int16, bps=1):
    flat = audio_int16.reshape(-1)
    return int(flat.size) * int(bps)

def prng_order(n, key: str):
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.permutation(n)

def embed_audio_stream(audio_int16, bits: np.ndarray, key: str, bps=1):
    audio = audio_int16.copy()
    flat = audio.reshape(-1)
    order = prng_order(len(flat), key)

    cap = len(flat) * int(bps)
    if len(bits) > cap:
        return None, f"Capacity exceeded. Need {len(bits)} bits, have {cap} bits."

    idx = 0
    bps = int(bps)
    for pos in order:
        u = int(flat[pos]) & 0xFFFF
        for bitpos in range(bps):
            if idx >= len(bits):
                break
            mask = 1 << bitpos
            u = (u & ~mask) | (int(bits[idx]) << bitpos)
            idx += 1
        if u >= 0x8000:
            u -= 0x10000
        flat[pos] = np.int16(u)
        if idx >= len(bits):
            break

    return audio.reshape(audio_int16.shape), "OK"

def extract_audio_stream(audio_int16, n_bits: int, key: str, bps=1):
    flat = audio_int16.reshape(-1)
    order = prng_order(len(flat), key)

    out = []
    bps = int(bps)
    for pos in order:
        u = int(flat[pos]) & 0xFFFF
        for bitpos in range(bps):
            out.append((u >> bitpos) & 1)
            if len(out) >= n_bits:
                return np.array(out, dtype=np.uint8), "OK"
    return None, "Not enough samples to extract requested bits."

def snr_db(a_int16, b_int16):
    a = a_int16.astype(np.float32)
    b = b_int16.astype(np.float32)
    noise = a - b
    p_signal = np.mean(a*a) + 1e-12
    p_noise = np.mean(noise*noise) + 1e-12
    return 10.0 * math.log10(p_signal / p_noise)

def eve_attack_audio(audio_int16, mode, strength):
    out = audio_int16.copy().astype(np.int32)

    if mode == "None":
        return audio_int16, "Forwarded as-is."

    if mode == "Noise":
        ns = int(strength) * 50
        noise = np.random.randint(-ns, ns + 1, size=out.shape)
        out = np.clip(out + noise, -32768, 32767).astype(np.int16)
        return out, f"Noise ±{ns}"

    if mode == "Quantize":
        q = max(2, int(strength))
        out = (out // q) * q
        out = np.clip(out, -32768, 32767).astype(np.int16)
        return out, f"Quantized step={q}"

    if mode == "Trim + Pad":
        cut = min(out.shape[0] // 10, int(strength) * 1000)
        if out.ndim == 1:
            trimmed = out[cut:]
            padded = np.pad(trimmed, (0, cut), mode="constant")
        else:
            trimmed = out[cut:, :]
            padded = np.pad(trimmed, ((0, cut), (0, 0)), mode="constant")
        return padded.astype(np.int16), f"Trimmed {cut} samples then padded"

    if mode == "Sample Tamper":
        flips = max(2000, int(strength) * 5000)
        flat = out.reshape(-1).astype(np.int16)
        idx = np.random.randint(0, flat.size, size=flips)
        u = (flat[idx].astype(np.int32) & 0xFFFF) ^ 1
        u[u >= 0x8000] -= 0x10000
        flat[idx] = u.astype(np.int16)
        return flat.reshape(out.shape).astype(np.int16), f"Flipped {flips} random sample LSBs"

    return audio_int16, "Unknown mode"

# =========================
# AUDIO SAFETY ANALYSIS
# =========================
def _to_mono(audio_int16: np.ndarray) -> np.ndarray:
    if audio_int16.ndim == 1:
        return audio_int16
    return (audio_int16.astype(np.int32).mean(axis=1)).astype(np.int16)

def audio_activity_score(audio_int16: np.ndarray, sr: int, win_ms=50):
    x = _to_mono(audio_int16).astype(np.float32)
    if sr <= 0:
        sr = 44100

    win = max(128, int(sr * (win_ms / 1000.0)))
    n = len(x)
    if n < win:
        x = np.pad(x, (0, win - n))
        n = len(x)

    m = n // win
    x = x[:m * win].reshape(m, win)

    rms = np.sqrt(np.mean((x / 32768.0) ** 2, axis=1) + 1e-12)

    s = np.sign(x)
    s[s == 0] = 1
    zc = np.mean(s[:, 1:] != s[:, :-1], axis=1).astype(np.float32)

    rms_n = (rms - rms.min()) / (rms.max() - rms.min() + 1e-12)
    zc_n = (zc - zc.min()) / (zc.max() - zc.min() + 1e-12)

    activity = 0.7 * rms_n + 0.3 * zc_n
    activity = np.clip(activity, 0, 1)

    return rms, zc, activity, win

def audio_segment_ranking(audio_int16: np.ndarray, sr: int, seg_ms=1000, topk=12):
    x = _to_mono(audio_int16).astype(np.float32)
    seg = max(256, int(sr * (seg_ms / 1000.0)))

    n = len(x)
    if n < seg:
        seg = n

    m = max(1, n // seg)
    scores = []
    for i in range(m):
        a = x[i * seg:(i + 1) * seg]
        if len(a) < 64:
            continue
        rms = float(np.sqrt(np.mean((a / 32768.0) ** 2) + 1e-12))
        diff = float(np.mean(np.abs(np.diff(a))) / 32768.0)
        score = 0.65 * rms + 0.35 * min(1.0, diff * 4.0)
        scores.append((i, score, rms, diff))

    table = [["Rank", "Segment#", "Start(s)", "Score", "RMS", "Diff"]]
    if not scores:
        return table

    scores.sort(key=lambda t: t[1], reverse=True)
    top = scores[:min(topk, len(scores))]
    for r, (i, sc, rms, diff) in enumerate(top, start=1):
        start_s = (i * seg) / float(sr)
        table.append([r, i, f"{start_s:.2f}", f"{sc:.4f}", f"{rms:.4f}", f"{diff:.4f}"])
    return table

def audio_safety_score(activity: np.ndarray):
    if activity is None or len(activity) == 0:
        return 0.0
    mean_a = float(np.mean(activity))
    low_frac = float(np.mean(activity < 0.25))
    score = (mean_a * 100.0) * (1.0 - 0.55 * low_frac)
    return float(np.clip(score, 0, 100))

def render_activity_bar(activity: np.ndarray, width=900, height=140):
    if activity is None or len(activity) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    a = np.clip(activity.astype(np.float32), 0, 1)
    x = np.linspace(0, len(a) - 1, width).astype(np.int64)
    a = a[x]
    base = (a * 255).astype(np.uint8)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 1] = base
    img[:, :, 2] = (base * 0.35).astype(np.uint8)
    img[:, :, 0] = (base * 0.15).astype(np.uint8)
    img[height // 2:height // 2 + 2, :, :] = 200
    return img

# =========================
# STATE + DOWNLOADS
# =========================
def new_state():
    return {
        # image
        "cover": None, "stego": None, "recv": None, "heat": None, "bpc": 2, "r": 3,
        # audio
        "audio_cover": None, "audio_stego": None, "audio_recv": None, "audio_sr": None,
        "audio_bps": 1, "audio_r": 3,
    }

def save_temp_png(img_u8: np.ndarray, prefix="img"):
    if img_u8 is None:
        return None
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".png")
    os.close(fd)
    Image.fromarray(img_u8).save(path)
    return path

def save_temp_wav(audio_int16: np.ndarray, sr: int, prefix="aud"):
    if audio_int16 is None or sr is None:
        return None
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(fd)
    write_wav(path, audio_int16, sr)
    return path

# =========================
# IMAGE CALLBACKS
# =========================
def do_embed(img, msg, key, bpc, ecc_r, st):
    try:
        if img is None:
            return st, None, None, None, None, None, "❌ Upload an image."
        if not msg:
            return st, None, None, None, None, None, "❌ Enter a message."
        if not key:
            return st, None, None, None, None, None, "❌ Enter a key."

        cover = resize_u8(ensure_u8_rgb(img), IMG_SIZE)
        heat = compute_heatmap(cover)

        bpc = int(bpc)
        r = int(ecc_r)

        header_bits, payload_bits = pack_message_bits(msg, key)
        header_rep = ecc_repeat(header_bits, r)
        payload_rep = ecc_repeat(payload_bits, r)

        stego, (hs, ps), ok = embed_bits(cover, header_rep, payload_rep, heat, bpc)
        if stego is None:
            return st, cover, None, None, None, None, f"❌ Embed failed: {ok}"

        st = new_state()
        st["cover"] = cover
        st["stego"] = stego
        st["recv"] = None
        st["heat"] = heat
        st["bpc"] = bpc
        st["r"] = r

        overlay = heat_overlay(cover, heat, strength=0.62)
        dmap = diff_map_u8(cover, stego)

        stego_file = save_temp_png(stego, "stego_")

        cap_bits = IMG_SIZE * IMG_SIZE * 3 * bpc
        used_bits = len(header_rep) + len(payload_rep)
        log = (f"✅ Embedded OK | PSNR: {psnr(cover, stego):.2f} dB\n"
               f"Header pixels: {hs} | Payload pixels: {ps}\n"
               f"ECC repetition r={r} | Used {used_bits} bits / Capacity {cap_bits} bits")

        return st, cover, stego, overlay, dmap, stego_file, log
    except Exception as e:
        return st, None, None, None, None, None, f"❌ Embed crashed: {e}"

def load_stego(st):
    if st.get("stego") is None:
        return None, "❌ Embed first."
    return st["stego"], "✅ Loaded stego."

def do_eve(st, mode, strength, crop_ratio):
    try:
        if st.get("stego") is None:
            return st, None, None, None, "❌ Embed first."

        attacked, log = eve_attack(st["stego"], mode, int(strength), float(crop_ratio))
        st["recv"] = attacked

        tmap = diff_map_u8(st["stego"], attacked)
        attacked_file = save_temp_png(attacked, "recv_")

        return st, attacked, tmap, attacked_file, f"⚠️ Eve: {log}"
    except Exception as e:
        return st, None, None, None, f"❌ Eve crashed: {e}"

def do_detect(st, key):
    try:
        if st.get("stego") is None:
            return "❌ No stego in system.", ""
        if not key:
            return "❌ Enter key.", ""

        recv = st["recv"] if st.get("recv") is not None else st["stego"]
        heat = st.get("heat")
        bpc = int(st.get("bpc", 2))
        r = int(st.get("r", 1))

        total_bits, msg = extract_bits(recv, heat, bpc, r)
        if total_bits is None:
            return f"❌ Extract failed: {msg}", ""

        ok, out = unpack_message_bits(total_bits, key)

        score = float(np.abs(st["stego"].astype(np.int16) - recv.astype(np.int16)).mean())
        status = "✅ SAFE (Integrity OK)" if ok else "❌ MODIFIED / WRONG KEY"
        details = f"{status}\nDiff score vs stego: {score:.3f}\n{('Message recovered.' if ok else out)}"

        return details, (out if ok else "")
    except Exception as e:
        return f"❌ Detect crashed: {e}", ""

# =========================
# AUDIO CALLBACKS
# =========================
def do_embed_audio(audio_file, msg, key, bps, ecc_r, st):
    try:
        if audio_file is None:
            return st, None, None, None, "❌ Upload a WAV file."
        if not msg:
            return st, None, None, None, "❌ Enter a message."
        if not key:
            return st, None, None, None, "❌ Enter a key."

        audio, sr = read_wav(audio_file)
        bps = int(bps)
        r = int(ecc_r)

        header_bits, payload_bits = pack_message_bits(msg, key)
        header_rep = ecc_repeat(header_bits, r)
        payload_rep = ecc_repeat(payload_bits, r)
        total_rep = np.concatenate([header_rep, payload_rep], axis=0)

        cap = audio_capacity_bits(audio, bps=bps)
        if len(total_rep) > cap:
            return st, None, None, None, f"❌ Capacity exceeded. Need {len(total_rep)} bits, have {cap} bits."

        stego_audio, ok = embed_audio_stream(audio, total_rep, key=f"AUD|{key}", bps=bps)
        if stego_audio is None:
            return st, None, None, None, f"❌ Embed failed: {ok}"

        st["audio_cover"] = audio
        st["audio_stego"] = stego_audio
        st["audio_recv"] = None
        st["audio_sr"] = sr
        st["audio_bps"] = bps
        st["audio_r"] = r

        cover_path = save_temp_wav(audio, sr, prefix="aud_cover_")
        stego_path = save_temp_wav(stego_audio, sr, prefix="aud_stego_")

        log = (
            f"✅ Audio embedded OK | SNR: {snr_db(audio, stego_audio):.2f} dB\n"
            f"bps={bps} | ECC r={r}\n"
            f"Used {len(total_rep)} bits / Capacity {cap} bits"
        )

        return st, cover_path, stego_path, stego_path, log
    except Exception as e:
        return st, None, None, None, f"❌ Audio embed crashed: {e}"

def load_audio_stego(st):
    if st.get("audio_stego") is None or st.get("audio_sr") is None:
        return None, "❌ Embed audio first."
    path = save_temp_wav(st["audio_stego"], st["audio_sr"], prefix="aud_loaded_")
    return path, "✅ Loaded stego audio."

def do_eve_audio(st, mode, strength):
    try:
        if st.get("audio_stego") is None or st.get("audio_sr") is None:
            return st, None, None, "❌ Embed audio first."

        attacked, log = eve_attack_audio(st["audio_stego"], mode, int(strength))
        st["audio_recv"] = attacked

        recv_path = save_temp_wav(attacked, st["audio_sr"], prefix="aud_recv_")

        diff = st["audio_stego"].astype(np.int32) - attacked.astype(np.int32)
        diff_score = float(np.mean(np.abs(diff)))

        msg = f"⚠️ Eve: {log}\nDiff score vs stego: {diff_score:.3f}"
        return st, recv_path, recv_path, msg
    except Exception as e:
        return st, None, None, f"❌ Audio Eve crashed: {e}"

def do_detect_audio(st, key):
    try:
        if st.get("audio_stego") is None or st.get("audio_sr") is None:
            return "❌ No audio stego in system.", ""
        if not key:
            return "❌ Enter key.", ""

        recv = st["audio_recv"] if st.get("audio_recv") is not None else st["audio_stego"]
        bps = int(st.get("audio_bps", 1))
        r = int(st.get("audio_r", 1))

        header_rep_len = HEADER_BITS * r
        header_rep_bits, ok = extract_audio_stream(recv, header_rep_len, key=f"AUD|{key}", bps=bps)
        if header_rep_bits is None:
            return f"❌ Extract header failed: {ok}", ""

        header_bits = ecc_majority(header_rep_bits, r)
        if len(header_bits) < 32:
            return "❌ Header decode failed.", ""

        L = bits_to_u32(header_bits[:32])
        if L <= 0:
            return "❌ Invalid payload length.", ""

        payload_rep_len = int(L) * r
        total_rep_len = header_rep_len + payload_rep_len

        total_rep, ok2 = extract_audio_stream(recv, total_rep_len, key=f"AUD|{key}", bps=bps)
        if total_rep is None:
            return f"❌ Extract failed: {ok2}", ""

        payload_rep = total_rep[header_rep_len:]
        payload_bits = ecc_majority(payload_rep, r)[:L]

        total_bits = np.concatenate([header_bits[:32], payload_bits], axis=0)
        okmsg, out = unpack_message_bits(total_bits, key)

        base = st["audio_stego"]
        diff = base.astype(np.int32) - recv.astype(np.int32)
        diff_score = float(np.mean(np.abs(diff)))

        status = "✅ SAFE (Integrity OK)" if okmsg else "❌ MODIFIED / WRONG KEY"
        details = f"{status}\nDiff score vs stego: {diff_score:.3f}\n{('Message recovered.' if okmsg else out)}"
        return details, (out if okmsg else "")
    except Exception as e:
        return f"❌ Audio detect crashed: {e}", ""

# =========================
# AUDIO ANALYSIS CALLBACK
# =========================
def do_audio_analysis(audio_path, win_ms, seg_ms):
    try:
        if audio_path is None:
            return None, [["Rank","Segment#","Start(s)","Score","RMS","Diff"]], "❌ Upload a WAV file."

        audio, sr = read_wav(audio_path)
        _, _, activity, _ = audio_activity_score(audio, sr, win_ms=int(win_ms))
        score = audio_safety_score(activity)
        table = audio_segment_ranking(audio, sr, seg_ms=int(seg_ms), topk=12)
        heatbar = render_activity_bar(activity)

        rec = []
        for row in table[1:6]:
            rec.append(f"segment {row[1]} @ {row[2]}s")
        rec_txt = ", ".join(rec) if rec else "N/A"

        txt = (
            f"✅ Audio safety analysis complete\n"
            f"Sample rate: {sr} Hz\n"
            f"Window: {int(win_ms)} ms | Segment: {int(seg_ms)} ms\n"
            f"Estimated safety score: {score:.1f}/100\n\n"
            f"How to use:\n"
            f"- Prefer high-activity regions (green heatbar)\n"
            f"- Avoid quiet/silent regions (dark)\n"
            f"- Recommended embed segments: {rec_txt}"
        )
        return heatbar, table, txt
    except Exception as e:
        return None, [["Rank","Segment#","Start(s)","Score","RMS","Diff"]], f"❌ Audio analysis crashed: {e}"

# =========================
# STYLE
# =========================
CSS = """
:root{
  --bg:#070b16;
  --panel: rgba(14, 24, 46, .72);
  --border: rgba(148,163,184,.18);
  --txt:#e5e7eb;
  --a:#7c3aed;
  --b:#22d3ee;
}
.gradio-container{ background:#070b16 !important; color:var(--txt) !important; }
.block, .gr-row, .gr-column{ position:relative; }
.block{
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,.30) !important;
}
button{ border-radius: 14px !important; font-weight: 800 !important; }
button.primary{
  border:none !important;
  background: linear-gradient(90deg, var(--a), var(--b)) !important;
}
#hud{
  padding: 16px 18px;
  border-radius: 18px;
  border: 1px solid rgba(148,163,184,.20);
  background: rgba(8, 14, 32, .72);
  box-shadow: 0 10px 30px rgba(0,0,0,.28);
  margin-bottom: 14px;
}
.smallmuted{ color: rgba(229,231,235,.75); font-size: 13px; }
"""

HUD = """
<div id="hud">
  <div style="font-weight:900;font-size:22px;letter-spacing:.3px;">
    SpyOps Stego Console <span style="color:#a5b4fc;">(StegoLab)</span>
  </div>
  <div class="smallmuted" style="margin-top:6px;">
    • Image steganography • Audio steganography • Audio safety analysis • Integrity checks • Eve attacks
  </div>
</div>
"""

# =========================
# UI
# =========================
with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    st = gr.State(new_state())
    gr.HTML(HUD)

    with gr.Tabs():

        # ================= IMAGE =================
        with gr.Tab("🖼️ Image Stego"):
            with gr.Tabs():

                with gr.Tab("1) Sender • Embed"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            cover_in = gr.Image(label="📷 Cover Image", type="numpy", height=260)
                            msg_in = gr.Textbox(label="✉️ Secret Message", placeholder="Type message…")
                            key_in = gr.Textbox(label="🔑 Key", type="password", placeholder="Secret key…")
                            bpc = gr.Slider(1, 2, step=1, value=2, label="Bits per channel (1 stealth, 2 capacity)")
                            ecc_r = gr.Slider(1, 7, step=2, value=3, label="ECC repetition r (1=off, 3/5/7 stronger)")
                            embed_btn = gr.Button("🔐 Embed Message", variant="primary")
                            log = gr.Textbox(label="System Log", lines=5)
                            stego_dl = gr.File(label="⬇️ Download Stego PNG")

                        with gr.Column(scale=1):
                            cover_out = gr.Image(label="Original (resized)", height=260)
                            stego_out = gr.Image(label="Stego Output", height=260)

                        with gr.Column(scale=1):
                            overlay_out = gr.Image(label="🔥 Heat Overlay (safer texture)", height=260)
                            diff_out = gr.Image(label="🧊 Change Map (cover vs stego)", height=260)

                    embed_btn.click(
                        do_embed,
                        inputs=[cover_in, msg_in, key_in, bpc, ecc_r, st],
                        outputs=[st, cover_out, stego_out, overlay_out, diff_out, stego_dl, log],
                    )

                with gr.Tab("2) Eve • Intercept & Attack"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            load_btn = gr.Button("📥 Load Stego From System", variant="primary")
                            eve_in = gr.Image(label="Intercepted Stego", height=260)

                            mode = gr.Dropdown(
                                ["None", "Noise", "Crop + Resize", "JPEG-like (Quantize)", "Pixel Tamper"],
                                value="None",
                                label="Attack Type"
                            )
                            strength = gr.Slider(1, 30, value=10, step=1, label="Strength")
                            crop_ratio = gr.Slider(0.50, 0.95, value=0.80, step=0.01, label="Crop ratio")
                            attack_btn = gr.Button("⚠️ Apply Attack & Forward", variant="primary")
                            eve_log = gr.Textbox(label="Eve Log", lines=3)
                            recv_dl = gr.File(label="⬇️ Download Received PNG")

                        with gr.Column(scale=1):
                            recv_out = gr.Image(label="Forwarded Image (to Receiver)", height=320)

                        with gr.Column(scale=1):
                            tamper_out = gr.Image(label="🚨 Tamper Map (stego vs received)", height=320)

                    load_btn.click(load_stego, inputs=[st], outputs=[eve_in, eve_log])
                    attack_btn.click(do_eve, inputs=[st, mode, strength, crop_ratio],
                                    outputs=[st, recv_out, tamper_out, recv_dl, eve_log])

                with gr.Tab("3) Detection + Receiver"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            key_recv = gr.Textbox(label="🔑 Key (Receiver)", type="password")
                            detect_btn = gr.Button("🔍 Detect + Decrypt", variant="primary")
                            det_out = gr.Textbox(label="Detection Result", lines=7)

                        with gr.Column(scale=1):
                            msg_out = gr.Textbox(label="Recovered Message", lines=10)

                    detect_btn.click(do_detect, inputs=[st, key_recv], outputs=[det_out, msg_out])

        # ================= AUDIO =================
        with gr.Tab("🎧 Audio Stego (WAV)"):
            with gr.Tabs():

                with gr.Tab("1) Sender • Embed"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            aud_in = gr.Audio(label="🎼 Cover Audio (WAV 16-bit)", type="filepath")
                            aud_msg = gr.Textbox(label="✉️ Secret Message", placeholder="Type message…")
                            aud_key = gr.Textbox(label="🔑 Key", type="password", placeholder="Secret key…")
                            aud_bps = gr.Slider(1, 2, step=1, value=1, label="Bits per sample (1 safest, 2 more capacity)")
                            aud_r = gr.Slider(1, 7, step=2, value=3, label="ECC repetition r (1=off, 3/5/7 stronger)")
                            aud_embed_btn = gr.Button("🔐 Embed in Audio", variant="primary")
                            aud_log = gr.Textbox(label="Audio Log", lines=5)
                            aud_dl = gr.File(label="⬇️ Download Stego WAV")

                        with gr.Column(scale=1):
                            aud_cover_play = gr.Audio(label="Original Audio", type="filepath")
                            aud_stego_play = gr.Audio(label="Stego Audio", type="filepath")

                    aud_embed_btn.click(
                        do_embed_audio,
                        inputs=[aud_in, aud_msg, aud_key, aud_bps, aud_r, st],
                        outputs=[st, aud_cover_play, aud_stego_play, aud_dl, aud_log],
                    )

                with gr.Tab("2) Eve • Intercept & Attack"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            aud_load_btn = gr.Button("📥 Load Stego Audio From System", variant="primary")
                            aud_eve_in = gr.Audio(label="Intercepted Stego Audio", type="filepath")

                            aud_mode = gr.Dropdown(
                                ["None", "Noise", "Quantize", "Trim + Pad", "Sample Tamper"],
                                value="None",
                                label="Attack Type"
                            )
                            aud_strength = gr.Slider(1, 30, value=10, step=1, label="Strength")
                            aud_attack_btn = gr.Button("⚠️ Apply Attack & Forward", variant="primary")
                            aud_eve_log = gr.Textbox(label="Eve Log", lines=4)
                            aud_recv_dl = gr.File(label="⬇️ Download Received WAV")

                        with gr.Column(scale=1):
                            aud_recv_play = gr.Audio(label="Forwarded Audio (to Receiver)", type="filepath")

                    aud_load_btn.click(load_audio_stego, inputs=[st], outputs=[aud_eve_in, aud_eve_log])
                    aud_attack_btn.click(
                        do_eve_audio,
                        inputs=[st, aud_mode, aud_strength],
                        outputs=[st, aud_recv_play, aud_recv_dl, aud_eve_log],
                    )

                with gr.Tab("3) Detection + Receiver"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            aud_key_recv = gr.Textbox(label="🔑 Key (Receiver)", type="password")
                            aud_detect_btn = gr.Button("🔍 Detect + Decrypt (Audio)", variant="primary")
                            aud_det_out = gr.Textbox(label="Detection Result", lines=7)

                        with gr.Column(scale=1):
                            aud_msg_out = gr.Textbox(label="Recovered Message", lines=10)

                    aud_detect_btn.click(do_detect_audio, inputs=[st, aud_key_recv], outputs=[aud_det_out, aud_msg_out])

                with gr.Tab("🧠 Safety Analysis (Audio)"):
                    gr.Markdown("**Green = safer** (busy/high-energy audio). **Dark = risky** (silence/flat).")
                    with gr.Row():
                        with gr.Column(scale=1):
                            aud_an_in = gr.Audio(label="Upload WAV for Analysis", type="filepath")
                            win_ms = gr.Slider(10, 200, value=50, step=10, label="Analysis window (ms)")
                            seg_ms = gr.Slider(250, 5000, value=1000, step=250, label="Segment size for ranking (ms)")
                            analyze_aud_btn = gr.Button("🧠 Analyze Audio Safety", variant="primary")
                            aud_safety_txt = gr.Textbox(label="Safety Report", lines=10)

                        with gr.Column(scale=1):
                            aud_activity_img = gr.Image(label="Activity Heatbar (green = safer)", height=180)

                    aud_rank_table = gr.Dataframe(
                        label="Top Safe Segments (embed here)",
                        headers=["Rank", "Segment#", "Start(s)", "Score", "RMS", "Diff"],
                        interactive=False,
                        row_count=13,
                        col_count=6
                    )

                    analyze_aud_btn.click(
                        do_audio_analysis,
                        inputs=[aud_an_in, win_ms, seg_ms],
                        outputs=[aud_activity_img, aud_rank_table, aud_safety_txt]
                    )

demo.launch(debug=True)
