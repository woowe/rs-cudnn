## How to build/run

# Install nvidia-docker

# Run this
```nvidia-docker build . -t $USER/rust-cudnn-gpu```
```nvidia-docker run -v $PWD:/root/Project -it $USER/rust-cudnn-gpu```