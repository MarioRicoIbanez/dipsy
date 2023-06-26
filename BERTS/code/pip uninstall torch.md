pip uninstall torch
pip uninstall torchvision torchtext torch-tensorrt 
pip install torch>=2.0
pip install git+"https://github.com/huggingface/transformers"
pip install git+"https://github.com/huggingface/accelerate"
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
pip install numba scipy -U