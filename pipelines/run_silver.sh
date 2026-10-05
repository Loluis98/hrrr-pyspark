#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/java-1.17.0-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export JAVA_TOOL_OPTIONS="-XX:+IgnoreUnrecognizedVMOptions -Dio.netty.tryReflectionSetAccessible=true --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"


source /home/luigi/miniconda/etc/profile.d/conda.sh
conda activate databricks

echo "Running Bronze -> Silver Pipeline with Arrow JVM Bypasses..."
python -m pipelines.bronze_to_silver
