TEMPLATE = subdirs
SUBDIRS =   solver_gpu solver_cpu application
QMAKE_CXXFLAGS += -mfma -mavx2 -m64 -mf16c -O3
OTHER_FILES += Common


