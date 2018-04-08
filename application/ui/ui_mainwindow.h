/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *s_mainWindowGridLayout;
    QSpacerItem *horizontalSpacer;
    QSpacerItem *horizontalSpacer_6;
    QSpacerItem *horizontalSpacer_2;
    QSpacerItem *horizontalSpacer_5;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QLabel *cellCountLabel;
    QSpinBox *cellCountSB;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_2;
    QRadioButton *cpuOpt;
    QRadioButton *gpuOpt;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_3;
    QRadioButton *brtOpt;
    QRadioButton *smrtOpt;
    QSpacerItem *verticalSpacer;
    QPushButton *genButt;
    QLineEdit *imageName;
    QPushButton *exportButt;
    QMenuBar *menubar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(800, 600);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        s_mainWindowGridLayout = new QGridLayout(centralwidget);
        s_mainWindowGridLayout->setObjectName(QStringLiteral("s_mainWindowGridLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer, 0, 4, 1, 1);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer_6, 0, 1, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer_2, 0, 2, 1, 1);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer_5, 0, 3, 1, 1);

        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        cellCountLabel = new QLabel(groupBox);
        cellCountLabel->setObjectName(QStringLiteral("cellCountLabel"));

        verticalLayout->addWidget(cellCountLabel);

        cellCountSB = new QSpinBox(groupBox);
        cellCountSB->setObjectName(QStringLiteral("cellCountSB"));
        cellCountSB->setMinimum(1);
        cellCountSB->setMaximum(1000);
        cellCountSB->setValue(20);

        verticalLayout->addWidget(cellCountSB);

        groupBox_2 = new QGroupBox(groupBox);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        verticalLayout_2 = new QVBoxLayout(groupBox_2);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        cpuOpt = new QRadioButton(groupBox_2);
        cpuOpt->setObjectName(QStringLiteral("cpuOpt"));
        cpuOpt->setChecked(true);

        verticalLayout_2->addWidget(cpuOpt);

        gpuOpt = new QRadioButton(groupBox_2);
        gpuOpt->setObjectName(QStringLiteral("gpuOpt"));
        gpuOpt->setEnabled(true);

        verticalLayout_2->addWidget(gpuOpt);


        verticalLayout->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(groupBox);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        verticalLayout_3 = new QVBoxLayout(groupBox_3);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        brtOpt = new QRadioButton(groupBox_3);
        brtOpt->setObjectName(QStringLiteral("brtOpt"));
        brtOpt->setChecked(true);

        verticalLayout_3->addWidget(brtOpt);

        smrtOpt = new QRadioButton(groupBox_3);
        smrtOpt->setObjectName(QStringLiteral("smrtOpt"));
        smrtOpt->setCheckable(false);

        verticalLayout_3->addWidget(smrtOpt);


        verticalLayout->addWidget(groupBox_3);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        genButt = new QPushButton(groupBox);
        genButt->setObjectName(QStringLiteral("genButt"));

        verticalLayout->addWidget(genButt);

        imageName = new QLineEdit(groupBox);
        imageName->setObjectName(QStringLiteral("imageName"));

        verticalLayout->addWidget(imageName);

        exportButt = new QPushButton(groupBox);
        exportButt->setObjectName(QStringLiteral("exportButt"));

        verticalLayout->addWidget(exportButt);


        s_mainWindowGridLayout->addWidget(groupBox, 0, 5, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 26));
        MainWindow->setMenuBar(menubar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Voronoi Generator", Q_NULLPTR));
        groupBox->setTitle(QApplication::translate("MainWindow", "Options", Q_NULLPTR));
        cellCountLabel->setText(QApplication::translate("MainWindow", "Cell Count", Q_NULLPTR));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "CPU/GPU", Q_NULLPTR));
        cpuOpt->setText(QApplication::translate("MainWindow", "CPU", Q_NULLPTR));
        gpuOpt->setText(QApplication::translate("MainWindow", "GPU", Q_NULLPTR));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "Brute/Smart", Q_NULLPTR));
        brtOpt->setText(QApplication::translate("MainWindow", "Brute", Q_NULLPTR));
        smrtOpt->setText(QApplication::translate("MainWindow", "Smart", Q_NULLPTR));
        genButt->setText(QApplication::translate("MainWindow", "Generate", Q_NULLPTR));
        imageName->setText(QApplication::translate("MainWindow", "imageName", Q_NULLPTR));
        exportButt->setText(QApplication::translate("MainWindow", "Export Image", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
