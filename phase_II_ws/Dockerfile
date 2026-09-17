FROM osrf/ros:jazzy-desktop
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-control \
    ros-jazzy-cv-bridge \
    ros-jazzy-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /phase_II_ws

COPY . .

RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

CMD ["bash"]
