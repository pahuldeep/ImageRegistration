TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
        registration.cpp

HEADERS += \
    registration.h

win32:CONFIG(release, debug|release): LIBS += -LC:/LIBRARY/OpenCV/x64/vc16/lib/ -lopencv_world480
else:win32:CONFIG(debug, debug|release): LIBS += -LC:/LIBRARY/OpenCV/x64/vc16/lib/ -lopencv_world480d

INCLUDEPATH += C:/LIBRARY/OpenCV/include
DEPENDPATH += C:/LIBRARY/OpenCV/include
