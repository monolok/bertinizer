from setuptools import setup, find_packages

setup(
    name='bertinizer',
    version='0.1',
    packages=find_packages(),
    description='A utility for fast EDA and plots',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas',
        'plotly',
        'scikit-learn'
    ],
    python_requires='>=3.6',
    author='Antoine Bertin',
    author_email='monolok35@gmail.com',
    url='https://github.com/yourusername/bertinizer',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Data-scientists',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='plotly pandas data-visualization EDA'
    )