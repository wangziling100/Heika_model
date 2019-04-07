FROM pytorch/pytorch
RUN conda install torchvision -c soumith
RUN conda install -c anaconda pandas
