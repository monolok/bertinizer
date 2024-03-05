from setuptools import setup, find_packages

setup(
    name='bertinizer',
    version='1.1',  # Incrementing the version to reflect the addition/update of functionality
    packages=find_packages(),
    description='A utility for fast EDA and plots, including features for outlier detection and handling categorical data in correlation analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas>=1.0.0',  # Specifying minimum versions can be a good practice for key dependencies
        'plotly>=4.0.0',
        'scikit-learn>=0.22',
        'numpy>=1.18.0'  # Adding numpy as a dependency if not already included, given its use in your code
    ],
    python_requires='>=3.6',  # Ensure compatibility with the versions used in your development
    author='Antoine Bertin',
    author_email='monolok35@gmail.com',
    url='https://github.com/monolok/bertinizer',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',  # Updated to reflect the progress in development
        'Intended Audience :: Data Scientists',  # Minor correction for clarity
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    keywords='plotly pandas data-visualization EDA outlier-detection'
)