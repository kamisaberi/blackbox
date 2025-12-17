package devops

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"blackbox-vacuum/internal/transport"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

type K8sCollector struct {
	KubeConfigPath string // Path to ~/.kube/config, or empty for In-Cluster
}

func (k *K8sCollector) Name() string {
	return "k8s_audit"
}

func (k *K8sCollector) Start(ctx context.Context, client *transport.CoreClient) {
	// 1. Authenticate (In-Cluster or Local)
	var config *rest.Config
	var err error

	if k.KubeConfigPath == "" {
		// Assume running inside the cluster
		config, err = rest.InClusterConfig()
	} else {
		// Use local kubeconfig file
		config, err = clientcmd.BuildConfigFromFlags("", k.KubeConfigPath)
	}

	if err != nil {
		log.Printf("[K8S] Auth Error: %v", err)
		return
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Printf("[K8S] Client Error: %v", err)
		return
	}

	log.Println("[K8S] Watching Events...")

	// 2. Watch Loop
	// We list all events in all namespaces
	for {
		select {
		case <-ctx.Done():
			return
		default:
			watcher, err := clientset.CoreV1().Events("").Watch(context.TODO(), metav1.ListOptions{})
			if err != nil {
				log.Printf("[K8S] Watch Error: %v. Retrying in 10s...", err)
				time.Sleep(10 * time.Second)
				continue
			}

			ch := watcher.ResultChan()
			for event := range ch {
				k8sEvent, ok := event.Object.(*v1.Event)
				if !ok {
					continue
				}

				// 3. Format Log
				logEntry := map[string]interface{}{
					"source":    "k8s",
					"reason":    k8sEvent.Reason,
					"message":   k8sEvent.Message,
					"obj_kind":  k8sEvent.InvolvedObject.Kind,
					"obj_name":  k8sEvent.InvolvedObject.Name,
					"namespace": k8sEvent.InvolvedObject.Namespace,
					"ts":        k8sEvent.LastTimestamp.Unix(),
				}

				jsonBytes, _ := json.Marshal(logEntry)
				payload := fmt.Sprintf("K8S: %s\n", string(jsonBytes))
				client.Send([]byte(payload))
			}
		}
	}
}