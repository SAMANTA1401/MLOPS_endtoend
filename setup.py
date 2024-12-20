from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e . '

# def get_requirement(file_path:str)->List[str] :
#         requirements = []
#         with open(file_path, 'r') as f:
#             requirements = f.readlines()
#             requirements = [req.replace("\n", " ") for req in requirements]
#             print(requirements)
#             print('xxxx')
#             print(HYPEN_E_DOT)

#         if HYPEN_E_DOT in requirements:
#             requirements.remove(HYPEN_E_DOT)
#             print(requirements)

#         return requirements 


# with open('README.md', 'r', encoding='utf-8') as f:
#     long_description = f.read()


__version__ = '0.0.1'
REPO_NAME = 'EndtoEnd_MLOps'
PKG_NAME = 'src'
AUTHOR_USER_NAME = 'SAMANTA1401'
AUTHOR_EMAIL = 'psamanta1401@gmail.com'

setup(
    name=PKG_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='A small package for mongodb connection',
    # long_description=long_description,
    # long_description_content='text/markdown',
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com//{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # extras_require={
    #     'dev': [
    #         # list your development requirements here
    #     ]
    # },
    # install_requires=get_requirement('requirements_dev.txt')
    # install_requires=['pymongo', 'pandas', 'numpy', 'python-dotenv', 'openpyxl','ensure','dnspython','pymongo[srv]']
)