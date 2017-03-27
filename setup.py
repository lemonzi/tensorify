from setuptools import setup

setup(name='tensorify',
      version='0.2.2',
      description='Decorate functions so they can be used in TensorFlow',
      url='http://github.com/lemonzi/tensorify',
      author='Quim Llimona',
      author_email='ql@lemonzi.me',
      license='MIT',
      packages=['tensorify'],
      install_requires=['tensorflow>=1.0'],
      zip_safe=True)
