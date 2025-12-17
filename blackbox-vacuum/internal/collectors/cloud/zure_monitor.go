package cloud

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"blackbox-vacuum/internal/transport"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/monitor/armmonitor"
)

type AzureCollector struct {
	SubscriptionID string
}

func (a *AzureCollector) Name() string {
	return "azure_monitor"
}

func (a *AzureCollector) Start(ctx context.Context, client *transport.CoreClient) {
	// 1. Authenticate (Uses ENV: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET)
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		log.Printf("[AZURE] Auth Error: %v", err)
		return
	}

	monitorClient, err := armmonitor.NewActivityLogsClient(a.SubscriptionID, cred, nil)
	if err != nil {
		log.Printf("[AZURE] Client Error: %v", err)
		return
	}

	log.Println("[AZURE] Polling Activity Logs...")

	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	// Look back 5 minutes
	lastTime := time.Now().Add(-5 * time.Minute)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			endTime := time.Now()
			
			// API Filter
			filter := fmt.Sprintf("eventTimestamp ge '%s' and eventTimestamp le '%s'", 
				lastTime.Format(time.RFC3339), 
				endTime.Format(time.RFC3339))

			pager := monitorClient.NewListPager(filter, nil)

			for pager.More() {
				page, err := pager.NextPage(ctx)
				if err != nil {
					log.Printf("[AZURE] Page Error: %v", err)
					break
				}

				for _, logItem := range page.Value {
					entry := map[string]interface{}{
						"source":    "azure",
						"operation": *logItem.OperationName.Value,
						"status":    *logItem.Status.Value,
						"resource":  *logItem.ResourceID,
						"ts":        logItem.EventTimestamp.Unix(),
					}
					
					// Get Caller Identity if available
					if logItem.Caller != nil {
						entry["caller"] = *logItem.Caller
					}

					jsonBytes, _ := json.Marshal(entry)
					payload := fmt.Sprintf("AZURE: %s\n", string(jsonBytes))
					client.Send([]byte(payload))
				}
			}
			lastTime = endTime
		}
	}
}