# Local Microsoft Phi-2

## Build & Run

- Clone the Repository:

```sh
git clone https://github.com/abughalib/phi2local
```

- Download the model, either the [Dolphin 2.0](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2) or [Microsoft Phi2](https://huggingface.co/microsoft/phi-2)

- Change the model path:
```rs
const MSPHI2_PATH_PREFIX: &'static str = "D:\\models\\phi-2";
const DOLPHIN_PATH_PREFIX: &'static str = "D:\\models\\dolphin_2.6";
```

- Build the code

```sh
cd phi2local; cargo build --release
```

- Run.

```sh
./target/release/phi2local
```

## Build Dependencies
- Install Rust from [Rustup.rs](https://rustup.rs/)

### Windows
- [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- [Clang Compiler](https://github.com/llvm/llvm-project/releases/)

### Debian based Linux
- Build Packages
```sh
sudo apt install build-essential clang
```