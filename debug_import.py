import sys
print("Python path:")
for path in sys.path:
    print(f"- {path}")

print("\nTrying to import fpdf2...")
try:
    import fpdf2
    print("fpdf2 imported successfully")
    print(f"fpdf2 version: {fpdf2.__version__}")
    print(f"fpdf2 location: {fpdf2.__file__}")
except ImportError as e:
    print(f"Import error: {e}")

print("\nChecking other dependencies...")
try:
    import pandas
    print("pandas imported successfully")
except ImportError as e:
    print(f"pandas import error: {e}")

try:
    import numpy
    print("numpy imported successfully")
except ImportError as e:
    print(f"numpy import error: {e}")

try:
    import sklearn
    print("sklearn imported successfully")
except ImportError as e:
    print(f"sklearn import error: {e}")
