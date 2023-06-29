from brainrender import Scene
from vedo import embedWindow, Plotter, show  # <- this will be used to render an embedded scene 
from itkwidgets import view

# embedWindow(None)  # <- this will make your scene popup

# popup_scene = Scene(title='popup')

# popup_scene.add_brain_region('MOs')

# popup_scene.render()  # press 'Esc' to close!!

embedWindow('k3d')  # <- this will make your scene embed with k3d

# Create a scene
scene = Scene(title='Embedded')  # note that the title will not actually display
scene.add_brain_region('MOs')

# Make sure it gets embedded in the window
scene.jupyter = True

# scene.render now will prepare the scene for rendering, but it won't render anything yet
scene.render()

#  to actually display the scene we use `vedo`'s `show` method to show the scene's actors
plt = Plotter()
plt.show(*scene.renderables)  # same as vedo.show(*scene.renderables)