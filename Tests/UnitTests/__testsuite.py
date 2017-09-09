import unittest
from Tests.UnitTests._BasePlot_UnitTest import _BasePlotTestCase
from Tests.UnitTests.DatasetTools_UnitTest import DatasetsToolsTestCase
from Tests.UnitTests.ModelUtils_UnitTest import ModelUtilsTestCase


def test_suite():
    """run all unittests at once"""
    suite = unittest.TestSuite()
    result = unittest.TestResult()
    suite.addTest(unittest.makeSuite(_BasePlotTestCase))
    suite.addTest(unittest.makeSuite(DatasetsToolsTestCase))
    suite.addTest(unittest.makeSuite(ModelUtilsTestCase))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))
    return suite


if __name__ == '__main__':
    test_suite()
