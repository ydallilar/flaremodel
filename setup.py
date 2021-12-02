from setuptools import setup, find_packages
from distutils.extension import Extension
import glob, platform

try:
    from Cython.Build import cythonize
except:
    raise Exception("Cython must be installed...")

try:
    import numpy as np
except:
    raise Exception("Numpy must be installed...")

WIN32= platform.system() == 'Windows'

# Temporary detection of GSL headers
def add_gsl_header_path(include_dirs):

    import subprocess as subp
    import warnings

    if not WIN32:
        try:
            proc = subp.check_output(["gsl-config", "--prefix"], shell=False)
            include_dirs.append("%s/include/" % proc.decode("utf-8").strip("\n"))
        except:
            warnings.warn("Can't locate GSL headers. Using system defaults.")
    
    return include_dirs


# Temporary OpenMP switch
def add_openmp_options(extra_link_args, extra_compile_args):

    import os

    if "FLAREMODEL_OPENMP" in os.environ:
        openmp_switch = False if os.environ["FLAREMODEL_OPENMP"] else True
    else:
        openmp_switch = True

    # Need to find relevant settings for different OSes.
    if openmp_switch:
        if not WIN32:
            extra_link_args.append("-fopenmp")
            extra_compile_args.append("-fopenmp")
        else:
            extra_link_args.append("/openmp")
            extra_compile_args.append("/openmp")


    return extra_link_args, extra_compile_args

include_dirs=[np.get_include(), "cfuncs"]
libraries=["gsl", "gslcblas"]
# Remove for the moement
#extra_compile_args=["-DHAVE_INLINE", "-march=native", "-std=c99"]
extra_compile_args=[]
extra_link_args=[]
ext_modules = {"name" : "flaremodel.utils.cfuncs", "sources" : ["flaremodel/utils/cfuncs.pyx", *glob.glob("cfuncs/*.c")]}

include_dirs = add_gsl_header_path(include_dirs)
extra_link_args, extra_compile_args = add_openmp_options(extra_link_args, extra_compile_args)

# Try a quick fix from galpy, honestly have no clue
if WIN32:
    extra_compile_args.append("-DGSL_DLL")
    extra_compile_args.append("-DWIN32")

cfuncs_module = Extension(ext_modules["name"], 
                        ext_modules["sources"], 
                        include_dirs=include_dirs, 
                        libraries=libraries,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

cmdclass = {}
try:
    from sphinx.setup_command import BuildDoc
    import shutil, os
    class BuildDocGHPages(BuildDoc):
        def run(self):
            super().run()

            for attr in self.user_options:
                if attr[0] == "build-dir=":
                    build_dir = attr[1]
                    break

            print(build_dir)
            build_dir = "build/sphinx/" if build_dir is None else build_dir
            if os.path.isdir("./docs/"): shutil.rmtree("./docs/")
            shutil.move("%s/html/" % build_dir, "./docs")
            open("./docs/.nojekyll", "w").close()

    cmdclass["build_sphinx_ghpages"] = BuildDocGHPages
except:
    pass
    
SPHINX_DEFAULTS = {'source_dir' : ('setup.py', 'sphinx')}

setup(
    name='flaremodel',
    version='1.0.3',
    author='Yigit Dallilar, Sebastiano von Fellenberg',
    author_email='ydallilar@mpe.mpg.de, sefe@mpe.mpg.de',
    packages = find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules = cythonize([cfuncs_module], emit_linenums=True),
    url="https://github.com/ydallilar/flaremodel/",
    description="A simple one zone code that can do many different flavours of flares",
    install_requires=["numpy", "scipy", "matplotlib", "lmfit>=1"],
    command_options={
        'build_sphinx' : SPHINX_DEFAULTS,
        'build_sphinx_ghpages' : SPHINX_DEFAULTS},
    cmdclass=cmdclass
) 
  
