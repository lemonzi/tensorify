from setuptools import setup

setup(name='tensorify',
      version='0.1',
      description='Decorate functions so they can be used in TensorFlow',
      url='http://github.com/lemonzi/tensorify',
      author='Quim Llimona',
      author_email='ql@lemonzi.me',
      license='MIT',
      packages=['tensorify'],
      install_requires=['tensorflow'],
      zip_safe=True)
