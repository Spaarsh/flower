## License

This code is licensed under the terms of the MIT license.

## Purpose

This directory contains proof-of-concept code used to study deserialization vulnerabilities for academic and defensive security research. The material is provided for educational use only and is intentionally written so that researchers can reproduce, study, and harden systems against these classes of vulnerabilities.

Important safety notice — do NOT run this code on production or public networks
--------------------------------------------------------------------------------

This repository contains code that demonstrates dangerous behaviors (for example, payloads that may execute code or establish remote connections). Running the code as-is can result in security compromises if executed on network-connected or production systems.

You must NOT run this code against any machines, services, or networks unless you explicitly own them or have written authorization to test them. Misuse can be illegal and may expose you or others to risk.

Intended audience
-----------------

- Security researchers and educators who will run experiments in isolated lab environments (virtual machines, containers, or air-gapped networks).
- Defensive engineers validating mitigations in private testbeds.

Permitted / forbidden uses
-------------------------

- Permitted: research, education, and testing inside environments you control or where you have explicit authorization.
- Forbidden: scanning, exploiting, or otherwise attacking systems or networks without explicit permission. Do not use these materials to attack third-party infrastructure.

Legal & institutional note
--------------------------

This project is distributed under the MIT license. A permissive license does not remove legal obligations. If you are affiliated with a company, university, or other organization, confirm that your tests comply with internal policies and applicable laws.


## Instructions
In order to use this modified flower package, build and install this by running the following in the root of this directory (`flower/`):
```bash
pip install -e .
```

Warning: example code in this directory contains unsafe proof-of-concept payloads that are disabled by default. To run those examples you must set the environment variable `ENABLE_UNSAFE_PAYLOAD=TRUE` in your shell and only run them inside an isolated lab environment (VMs/containers with no access to production or the public internet):

```bash
export ENABLE_UNSAFE_PAYLOAD=TRUE
```

Do not enable this on production or public networks.

To run a normal client, simply remove the __reduce__ method from the `Net class`.