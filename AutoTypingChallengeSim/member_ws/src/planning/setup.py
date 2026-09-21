from glob import glob

from setuptools import find_packages, setup

package_name = 'planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eqmem',
    maintainer_email='eqm.emoore@hotmail.com',
    description='Kinematics and IK for the 5-DOF typing arm',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kinematics = planning.nodes.kinematics_node:main',
            'type_keys = planning.nodes.key_sequence_node:main',
        ],
    },
)
