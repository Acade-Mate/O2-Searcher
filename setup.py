from setuptools import setup, find_packages

setup(
    name='o2searcher',
    version='0.0.0',
    description='Open-source training recipe for reproducing Deepseek-R1.',
    author='Agentica Team',
    packages=find_packages(include=['o2searcher',]),
    install_requires=[
        'google-cloud-aiplatform',
        'latex2sympy2',
        'pylatexenc',
        'sentence_transformers',
        'tabulate',
        'flash_attn==2.7.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)