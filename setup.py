from setuptools import setup

setup(
    name='revide_dataset',
    version='0.1.0',    
    description='This pulls in the REVIDE dataset',
    url='https://github.com/hannahkamundson/REVIDE-utils',
    author='Hannah Amundson',
    license='MIT',
    packages=['revide_dataset'],
    install_requires=['numpy',
                      'torch',
                      'imageio',
                      'opencv-python'    
                      ],
)