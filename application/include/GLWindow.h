#ifndef NGLSCENE_H_
#define NGLSCENE_H_


#include "Shader.h"
#include "TrackballCamera.h"
#include "Mesh.h"
#include "utils.h"

#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <ext.hpp>
#include <glm.hpp>
#include <QOpenGLWidget>
#include <QResizeEvent>
#include <QEvent>
#include <memory>
#include "../solver_cpu/include/SerialSolver.h"
#include "../solver_gpu/include/CudaSolver.h"
#include <QImage>


class GLWindow : public QOpenGLWidget
{
    Q_OBJECT // must include this if you use Qt signals/slots
public :
    /// @brief Constructor for GLWindow
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Constructor for GLWindow
    /// @param [in] _parent the parent window to create the GL context in
    //----------------------------------------------------------------------------------------------------------------------
    GLWindow( QWidget *_parent );

    /// @brief dtor
    ~GLWindow();
    void mouseDoubleClickEvent(QMouseEvent * _event);
    void mouseMove( QMouseEvent * _event );
    void mouseClick( QMouseEvent * _event );


    uint getCellCount() const;

    vec2 getDimensions();

    //----------------------------------------------------------------------------------------------------------------------
    SerialSolver m_CPUsolver;
    CUDASolver m_GPUsolver;

    bool getUsingCPU() const;




public slots:

    void init();
    void reset();
    void setCellCount(int cellCount);
    void updateDiagram();
    void exportImage();
    void setUsingCPU(bool _usingCPU);
    void setBrute(bool brute);
    void setImageSize(bool _1k);

protected:
    /// @brief  The following methods must be implimented in the sub class
    /// this is called when the window is created
    void initializeGL();

    /// @brief this is the main gl drawing routine which is called whenever the window needs to be re-drawn
    void paintGL();
    void renderScene();
    void renderTexture();

    void putPixel(vec3 _colour, uint _x, uint _y);

private :
    //----------------------------------------------------------------------------------------------------------------------
    Mesh * m_mesh;
    //----------------------------------------------------------------------------------------------------------------------
    std::array<Mesh, 1> m_meshes;
    //----------------------------------------------------------------------------------------------------------------------
    Shader m_shader;
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_vao;
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_vbo;
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_nbo;
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_tbo;
    //----------------------------------------------------------------------------------------------------------------------
    GLint m_vertexPositionAddress;
    //----------------------------------------------------------------------------------------------------------------------
    GLint m_vertexNormalAddress;
    //----------------------------------------------------------------------------------------------------------------------
    GLint m_MVAddress;
    //----------------------------------------------------------------------------------------------------------------------
    GLint m_MVPAddress;
    //----------------------------------------------------------------------------------------------------------------------
    GLint m_NAddress;
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_projection;
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_view;
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_MV;
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_MVP;
    //----------------------------------------------------------------------------------------------------------------------
    TrackballCamera m_camera;
    //----------------------------------------------------------------------------------------------------------------------
    int m_amountVertexData;
    //----------------------------------------------------------------------------------------------------------------------
    QImage m_image;
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<GLuint> m_textures;
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_colourTextureAddress;
    //----------------------------------------------------------------------------------------------------------------------
    void addTexture();
    //----------------------------------------------------------------------------------------------------------------------
    int prevX;
    //----------------------------------------------------------------------------------------------------------------------
    int prevY;
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<vec3> pixels;
    //----------------------------------------------------------------------------------------------------------------------
    uint m_cellCount = 20;

    bool m_usingCPU = true;

    bool m_brute = true;

    uint m_imageSize = 1024;

    bool m_updatedDiagram = false;
};

#endif
