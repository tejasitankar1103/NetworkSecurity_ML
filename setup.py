from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #read lines 
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                #ignore empty lines and -e .
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("Requirement.txt file not found")

    return requirement_lst

# print(get_requirements())

setup(
    name = "NetworkSecurity",
    version="0.0.1",
    author="Tejas Itankar",
    author_email="tejasitankar10@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()
)