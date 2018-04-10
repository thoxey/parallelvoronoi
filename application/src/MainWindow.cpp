#include "MainWindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    m_ui(new Ui::MainWindow)
{
    m_ui -> setupUi(this);
    m_ui -> setupUi(this);
    m_gl = new GLWindow(this);
    m_ui -> s_mainWindowGridLayout -> addWidget(m_gl,0,0,3,5);

    //QString imageName=m_ui->imageName->text();

    connect(m_ui->exportButt, SIGNAL(clicked(bool)), m_gl, SLOT(exportImage()));

    connect( m_ui->genButt, SIGNAL(clicked(bool)), m_gl, SLOT(updateDiagram()));

    connect( m_ui->cellCountSB,SIGNAL(valueChanged( int )), m_gl, SLOT(setCellCount(int)));

    connect(m_ui->cpuOpt,SIGNAL(toggled(bool)), m_gl, SLOT(setUsingCPU(bool)));

    connect(m_ui->brtOpt,SIGNAL(toggled(bool)), m_gl, SLOT(setBrute(bool)));

    connect(m_ui->optOneK,SIGNAL(toggled(bool)), m_gl, SLOT(setImageSize(bool)));
}

MainWindow::~MainWindow()
{
    delete m_ui;
}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::keyPressEvent(QKeyEvent *_event)
{
  // this method is called every time the main window recives a key event.
  // we then switch on the key value and set the camera in the GLWindow
  switch ( _event->key() )
  {
    case Qt::Key_Escape : QApplication::exit(EXIT_SUCCESS); break;
    case Qt::Key_Space : m_gl->m_CPUsolver.makeDiagram(m_gl->getDimensions(), m_gl->getCellCount());
    default : break;
  }
}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::mouseMoveEvent(QMouseEvent * _event)
{
  m_gl->mouseMove(_event);
}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::mousePressEvent(QMouseEvent * _event)
{
  m_gl->mouseClick(_event);

}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::mouseReleaseEvent(QMouseEvent * _event)
{
  m_gl->mouseClick(_event);
}

//----------------------------------------------------------------------------------------------------------------------
