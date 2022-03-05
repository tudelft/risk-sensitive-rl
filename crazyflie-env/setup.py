from setuptools import setup

setup(
    name='crazyflie_env',
    version='0.1',
    description='Crazyflie simulation environment based on OpenAI gym',
    author='Cheng Liu',
    packages=['crazyflie_env'],
    install_requires=['gym', 'numpy']
)