from setuptools import setup, find_packages


setup(
    name='EXOSpy',
    version='2.3',
    license='MIT',
    author="Gonzalo Cucho-Padin",
    author_email='gonzaloaugusto.cuchopadin@nasa.gov',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/gcucho/EXOSpy',
    keywords='exosphere FUV Lyman-alpha',
    install_requires=[
          'scikit-learn','numpy','scipy','matplotlib'
      ],

)
