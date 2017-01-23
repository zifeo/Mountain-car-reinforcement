### Plot functions

def vec_plot(p, ngrid_pos, ngrid_speed):
    '''Give an array `p` of probabilities, plots the Q-values direction vector field.'''
    p_max = np.argmax(p, axis=2)
    
    # define arrow direction
    U = (p_max == 0) * 1 + (p_max == 1) * -1
    V = np.zeros((ngrid_pos, ngrid_speed))

    plt.quiver(U, V, alpha=1, scale=1.8, units='xy')
    plt.xlim(-1, 20)
    plt.xticks(())
    plt.ylim(-1, 20)
    plt.yticks(())
    plt.xlabel('position $x$')
    plt.ylabel('speed $\dot x$')
    plt.title('Q-values direction vector field (arrows show the direction of applied force)')

    plt.show()

def plot3D(q, x_pos, y_speed):
    '''Given q-values `q`, plots in 3D all possibles states.'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # generate all parameters combination for states
    x, y = np.meshgrid(x_pos, y_speed)
    ax.plot_wireframe(x, y, q, color='grey')
    ax.set_xlabel('position')
    ax.set_ylabel('speed')
    ax.set_zlabel('max q')
    
    plt.show()

