from tests.tests import *
from tests import test_walker
from tests import test_shooting
from tests import test_mppi
from tests import test_shooting_params
import os

def main():
    test_mppi.main()

if __name__ == "__main__":
    # test_environment_functionality()
    test_mppi.main()