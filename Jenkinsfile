pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'ljs3536/shoppingmall-flask'
    }

    stages {
        stage('Checkout') {
            steps {
                git credentialsId: 'github', url: 'https://github.com/ljs3536/shoppingmall-flask-analysis.git'
            }
        }

        stage('Docker Build & Push') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', 'dockerhub') {
                        def customImage = docker.build("${DOCKER_IMAGE}")
                        customImage.push("latest")
                    }
                }
            }
        }
    }
}
