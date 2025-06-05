import sys
print("Python path:")
for path in sys.path:
    print(f"- {path}")

print("\nTrying to import fpdf...")
try:
    from fpdf import FPDF
    print("fpdf imported successfully")
    print(f"FPDF location: {FPDF.__module__}")
except ImportError as e:
    print(f"fpdf import error: {e}")

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
