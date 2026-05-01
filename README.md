# Option Pricing Quantum Framework
A Production Grade Framework For Option Pricing Using Quantum Computers

![](https://automatski.com/wp-content/uploads/2025/05/Automatski-New-Logo.svg)

![](https://raw.githubusercontent.com/adityayadav76/option_pricing_quantum_framework/refs/heads/main/screenshot.png)


## About

This repository contains an implementation of Option Pricing using Quantum Computers created by [Automatski](https://automatski.com). Automatski's Quantum Computers support upto 100's or Logical Qubits and millions of Gates.

It supports the following types of Options:

	European - Option that may only be exercised on expiry.

	American - Option that may be exercised on any trading day on or before expiration.

	Bermudan - Option that may be exercised only on specified dates on or before expiration.

	Asian -	Option whose payoff is determined by the average underlying price over some preset time period.

	Barrier - 	Option with the general characteristic that the underlying security's price must pass a certain level before it can be exercised.

	Binary - All-or-nothing option that pays the full amount if the underlying security meets the defined condition on expiration otherwise it expires.

  ***Omitted: Exotic - Any of a broad category of options that may include complex financial structures.

Pricing of:

	Individual options of the above six types

	A portfolio of options of the above six types


To see a demonstration of code in this repo please see the [Demo Video](https://youtu.be/OpWgkwLNl-8)

### Intellectual Property

All rights are reserved by Automatski for Automatski-authored components of this codebase. Rights to third-party or upstream components remain with their respective original authors and licensors.

## Installation

```sh
pip install -r requirements.txt
```

## Execution

On Windows run the batch file. On Linux/Mac see below

```sh
python app.py
```

# Automatski's Quantum SDKs

[Quantum Annealing SDK](https://bit.ly/4ej7yaw)

[Getting Started With Quantum Annealing Video](https://youtu.be/-wKqcIKxY0A)


[Quantum Computing SDK](https://bit.ly/3XU7NDX)

[Getting Started With Quantum Computing Video](https://youtu.be/o4x0YWJ4YMw)
