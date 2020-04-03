if [ -n "$TRAVIS_TAG" ]; then # only run conda-build on tags, sometimes takes too long
    echo Building conda package...

    conda create -n builder conda-build anaconda-client
    source activate builder

    #conda install conda-build anaconda-client
    #conda update -q conda-build
    #conda info -a

    # Build conda packages
    mkdir ../artifacts
    export ARTIFACTS_FOLDER=`readlink -f ../artifacts`
    
    conda config --set anaconda_upload no
    conda config --add channels conda-forge
    conda config --add channels nilmtk
    
    # Replace version with the tag
    sed -i "s/0\.4\.0\.dev1/$TRAVIS_TAG/g" conda/meta.yaml
    
    conda-build --quiet --no-test --output-folder "$ARTIFACTS_FOLDER" conda 

    if [ -n "$ANACONDA_API_TOKEN" ]; then 
        # Upload artifacts to anaconda.org
        ANACONDA_LABEL='main'
        if [[ $TRAVIS_TAG == *"dev"* ]]; then
            ANACONDA_LABEL='dev'
        fi
        find ../artifacts -name "*.whl" -or -name "*.tar.bz2" -or -name "*.conda" | xargs -I {} anaconda upload --no-progress -l $ANACONDA_LABEL -u nilmtk {}
    fi
fi
