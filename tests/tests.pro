include(../Common/common.pri)

TARGET = pvTest
CONFIG += console c++11
CONFIG -= app_bundle
QT += widgets testlib

OTHER_FILES += Common

OBJECTS_DIR = $$PWD/obj

INCLUDEPATH+= include /usr/local/include /public/devel/include

LIBS+= -L$$LIB_INSTALL_DIR -lsolver_cpu -lsolver_gpu \

INCLUDEPATH+= $$PWD/../Common/include \
              $$INC_INSTALL_DIR \

macx:CONFIG-=app_bundle

QMAKE_RPATHDIR += $$LIB_INSTALL_DIR

HEADERS += \
    include/mockcpusolver.h \
    include/mockgpusolver.h

SOURCES += \
    src/mockcpusolver.cpp \
    src/mockgpusolver.cpp \
    src/tests.cpp
