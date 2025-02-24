from setuptools import setup, find_packages

setup(
    name='meshgraph',
    version='0.1',
    packages=find_packages(),
    author='PhD Mikel M. Iparraguirre',
    author_email='mikel.martinez@unizar.es',
    description='Library that contains MeshGraph algorithm from DeepMind in Pytorch for Plasticity collisions',
    long_description_content_type='text/markdown',
    url='Plasticity-MeshGraphNets',
    install_requires=[
        # Add any dependencies your library needs
    ],
    python_requires='>=3.9',
)
