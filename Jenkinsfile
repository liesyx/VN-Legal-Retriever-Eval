// legal-rag API — GitHub push -> Jenkins: quét + test + build -> quét image -> Harbor -> Helm deploy lên k3s
// Mọi công cụ quét chạy bằng `docker run` trên DinD; mã nguồn đưa vào container qua stdin (tar), không cần chia sẻ volume.
pipeline {
  agent any

  environment {
    REGISTRY   = 'harbor.oci.lexsy.pro.vn'
    IMAGE_NAME = 'legal-rag/api'              // <harbor-project>/<repo>
    NAMESPACE  = 'legal-rag'
    RELEASE    = 'legal-rag'
    CHART_DIR  = 'helm/legal-rag'
    OBS_RELEASE   = 'obs'
    OBS_NAMESPACE = 'observability'

    // Ảnh công cụ: nên mirror vào Harbor (proxy cache) rồi ghim @sha256 — xem README
    GITLEAKS = 'zricethezav/gitleaks:v8.21.2'
    HADOLINT = 'hadolint/hadolint:v2.12.0'
    TRIVY    = 'aquasec/trivy:0.57.1'
    TRIVY_RUN = 'docker run --rm -v trivy-cache:/root/.cache'   // cache CSDL lỗ hổng trong volume của DinD

    // 1 = lỗi quét làm hỏng build (prod) · 0 = chỉ báo cáo (dev)
    SCAN_EXIT = '0'

    DOCKER_BUILDKIT = '1'
  }

  options {
    timestamps()
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '20'))
    timeout(time: 60, unit: 'MINUTES')
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
        script {
          env.IMAGE     = "${env.REGISTRY}/${env.IMAGE_NAME}"
          env.WEB_IMAGE = "${env.REGISTRY}/legal-rag/web"
          env.IMAGE_TAG = "${env.BUILD_NUMBER}-${sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()}"
          env.IS_MAIN   = (env.BRANCH_NAME == 'main' || env.GIT_BRANCH in ['main', 'origin/main']).toString()
        }
        echo "Image: ${env.IMAGE}:${env.IMAGE_TAG} · deploy: ${env.IS_MAIN}"
      }
    }

    stage('Static checks') {
      parallel {
        stage('Secrets (gitleaks)') {
          steps {
            // quét cả lịch sử git: key lỡ commit rồi xoá vẫn bị bắt
            sh '''
              tar -c . | docker run --rm -i --entrypoint sh $GITLEAKS -c \
                'mkdir /tmp/src && tar -x -C /tmp/src && cd /tmp/src &&
                 gitleaks git . --redact --no-banner --verbose --exit-code '"$SCAN_EXIT"
            '''
          }
        }
        stage('Dockerfile (hadolint)') {
          steps {
            sh '''
              docker run --rm -i $HADOLINT hadolint --failure-threshold warning - < Dockerfile
              docker run --rm -i $HADOLINT hadolint --failure-threshold warning - < web/Dockerfile
            '''
          }
        }
        stage('Helm + misconfig (trivy config)') {
          steps {
            sh '''
              helm lint $CHART_DIR --set image.tag=lint
              helm lint helm/observability
              tar -c Dockerfile $CHART_DIR | $TRIVY_RUN -i --entrypoint sh $TRIVY -c \
                'mkdir /tmp/src && tar -x -C /tmp/src &&
                 trivy config --quiet --exit-code '"$SCAN_EXIT"' --severity HIGH,CRITICAL --helm-set image.tag=scan /tmp/src'
            '''
          }
        }
      }
    }

    stage('Test') {
      steps {
        // stage "test" của Dockerfile: test logic (Qdrant trong bộ nhớ, model giả) + import torch/sentence-transformers thật
        sh 'docker build --target test -t $IMAGE:test-$IMAGE_TAG .'
      }
    }

    stage('Build image') {
      steps {
        sh '''
          docker build --target runtime \
            --label org.opencontainers.image.revision=$(git rev-parse HEAD) \
            --label org.opencontainers.image.version=$IMAGE_TAG \
            -t $IMAGE:$IMAGE_TAG .
          docker build -t $WEB_IMAGE:$IMAGE_TAG web
        '''
      }
    }

    stage('Scan image (trivy)') {
      steps {
        // Chặn khi có lỗ hổng CRITICAL đã có bản vá; HIGH chỉ báo cáo. SBOM CycloneDX lưu kèm build.
        sh '''
          $TRIVY_RUN -v /var/run/docker.sock:/var/run/docker.sock $TRIVY image --quiet \
            --scanners vuln,secret --severity HIGH,CRITICAL --ignore-unfixed --format table $IMAGE:$IMAGE_TAG
          $TRIVY_RUN -v /var/run/docker.sock:/var/run/docker.sock $TRIVY image --quiet \
            --scanners vuln --severity CRITICAL --ignore-unfixed --exit-code $SCAN_EXIT $IMAGE:$IMAGE_TAG
          $TRIVY_RUN -v /var/run/docker.sock:/var/run/docker.sock $TRIVY image --quiet \
            --format cyclonedx $IMAGE:$IMAGE_TAG > sbom.cdx.json
          $TRIVY_RUN -v /var/run/docker.sock:/var/run/docker.sock $TRIVY image --quiet \
            --scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code $SCAN_EXIT $WEB_IMAGE:$IMAGE_TAG
        '''
      }
      post { always { archiveArtifacts artifacts: 'sbom.cdx.json', allowEmptyArchive: true } }
    }

    stage('Push to Harbor') {
      when { expression { env.IS_MAIN == 'true' } }
      steps {
        withCredentials([usernamePassword(credentialsId: 'harbor-legal-rag', usernameVariable: 'REG_USER', passwordVariable: 'REG_PASS')]) {
          sh '''
            echo "$REG_PASS" | docker login $REGISTRY -u "$REG_USER" --password-stdin
            docker push $IMAGE:$IMAGE_TAG
            docker push $WEB_IMAGE:$IMAGE_TAG
            docker logout $REGISTRY
          '''
        }
      }
    }

    stage('Deploy observability') {
      // Jaeger · Prometheus · Loki · Grafana (namespace observability). Chart không đổi -> helm gần như không làm gì.
      // Lỗi ở đây chỉ đánh dấu UNSTABLE, không chặn deploy API.
      when { expression { env.IS_MAIN == 'true' } }
      steps {
        catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
          withCredentials([file(credentialsId: 'kubeconfig-k3s', variable: 'KUBECONFIG')]) {
            sh '''
              helm upgrade --install $OBS_RELEASE helm/observability \
                --namespace $OBS_NAMESPACE --atomic --wait --timeout 5m --history-max 5
            '''
          }
        }
      }
    }

    stage('Deploy (Helm -> k3s)') {
      when { expression { env.IS_MAIN == 'true' } }
      steps {
        withCredentials([file(credentialsId: 'kubeconfig-k3s', variable: 'KUBECONFIG')]) {
          sh '''
            helm upgrade --install $RELEASE $CHART_DIR \
              --namespace $NAMESPACE \
              --set image.repository=$IMAGE --set image.tag=$IMAGE_TAG \
              --atomic --wait --timeout 15m --history-max 10
            helm -n $NAMESPACE history $RELEASE --max 5
          '''
        }
      }
    }
  }

  post {
    always {
      sh 'docker rmi $IMAGE:test-$IMAGE_TAG $IMAGE:$IMAGE_TAG $WEB_IMAGE:$IMAGE_TAG 2>/dev/null || true; docker image prune -f >/dev/null 2>&1 || true'
    }
    success { echo "✅ ${env.IMAGE}:${env.IMAGE_TAG}" }
    failure { echo '❌ Lỗi. Nếu hỏng ở bước deploy, --atomic đã tự rollback về bản trước.' }
  }
}
