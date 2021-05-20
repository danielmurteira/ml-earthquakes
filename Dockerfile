FROM python:slim

RUN pip install -U scikit-learn
RUN pip install numpy matplotlib scipy pandas Pillow imageio

COPY . /home
CMD cd /home && python earthquakes.py