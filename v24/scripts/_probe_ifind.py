import sys, traceback
mods = ['ifind','iFinD','THS','ifind_api']
for mod in mods:
    try:
        m = __import__(mod)
        print("IMPORT_OK", mod, getattr(m,'__file__','?'))
    except Exception as e:
        print("IMPORT_FAIL", mod, repr(e)[:100])
# win32com iFinD (同花顺 iFinD 常以 COM 暴露)
try:
    import win32com.client
    print("win32com_available")
except Exception as e:
    print("win32com_fail", repr(e)[:80])
