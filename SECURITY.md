# Security Notes

StegoLab is an **educational and experimental** steganography application.

## What it provides

- hidden-message embedding in images and PCM WAV audio
- a payload marker
- SHA-256 integrity checking after recovery
- repetition-code error correction
- key-dependent byte obfuscation / embedding position selection
- attack simulation for learning and robustness experiments

## What it does not provide

The XOR operation in this project is **not cryptographic encryption**. Repeating a user-supplied key over payload bytes does not provide confidentiality against a capable attacker.

Likewise, an unkeyed SHA-256 digest provides integrity checking of the recovered message but is not a cryptographic message-authentication code.

Do not use this project to protect:

- passwords
- financial information
- private keys
- medical records
- confidential business information
- regulated or safety-critical data

## Production-grade direction

A security-focused redesign should consider:

- AES-GCM or ChaCha20-Poly1305 authenticated encryption
- a modern KDF such as Argon2id, scrypt, or PBKDF2 when deriving keys from passwords
- authenticated metadata
- strict file-size/input validation
- resource limits
- dependency scanning
- fuzz/property testing of decoders

## Reporting

If you discover a security issue in the project itself, avoid publishing real secrets or sensitive test data in a public issue.
