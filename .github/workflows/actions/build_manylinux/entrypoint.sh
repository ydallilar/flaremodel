#!/bin/sh

yum -y install gsl-devel

PYVER=${1#.}

case $PYVER in

	35 | 36 | 37)
		PYTHON_VERSION=cp${PYVER}-cp${PYVER}m
		;;
	38 | 39 | 310)
		PYTHON_VERSION=cp${PYVER}-cp${PYVER}
		;;
	*)
		echo OOPS
		;;
esac

/opt/python/${PYTHON_VERSION}/bin/pip install --upgrade pip
/opt/python/${PYTHON_VERSION}/bin/pip install -U wheel auditwheel
/opt/python/${PYTHON_VERSION}/bin/pip install -U cython numpy
/opt/python/${PYTHON_VERSION}/bin/python setup.py bdist_wheel -d wheelhouse

auditwheel repair /github/workspace/wheelhouse/*.whl
rm /github/workspace/wheelhouse/*.whl



