#!/bin/sh

echo $1
yum -y install gsl-devel

PYVER=${1/./}
echo ${PYVER}

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

echo ${PYTHON_VERSION}

/opt/python/${PYTHON_VERSION}/bin/pip install --upgrade pip
/opt/python/${PYTHON_VERSION}/bin/pip install -U wheel auditwheel
/opt/python/${PYTHON_VERSION}/bin/pip install -U cython "numpy<1.20"
/opt/python/${PYTHON_VERSION}/bin/python setup.py bdist_wheel -d wheelhouse

WHEEL=$(ls wheelhouse/*.whl)
auditwheel repair $WHEEL
rm $WHEEL

