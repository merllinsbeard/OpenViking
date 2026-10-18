// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0

package openviking

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"reflect"
	"testing"
)

func originFixture(t *testing.T) (Message, map[string]any) {
	t.Helper()
	raw, err := os.ReadFile("../../tests/fixtures/message-origin-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Message  Message `json:"message"`
		Response struct {
			Result map[string]any `json:"result"`
		} `json:"response"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture.Message, fixture.Response.Result
}

func TestOriginNativeSDKSingleAndBatchReadback(t *testing.T) {
	for _, batch := range []bool{false, true} {
		t.Run(map[bool]string{false: "single", true: "batch"}[batch], func(t *testing.T) {
			message, response := originFixture(t)
			calls := 0
			client, closeServer := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var request map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				var sent Message
				if batch {
					var messages []Message
					if err := json.Unmarshal(request["messages"], &messages); err != nil {
						t.Error(err)
					}
					if len(messages) != 1 {
						t.Errorf("messages=%d", len(messages))
						return
					}
					sent = messages[0]
				} else {
					if err := json.Unmarshal(request["origin"], &sent.Origin); err != nil {
						t.Error(err)
					}
				}
				if !reflect.DeepEqual(sent.Origin, message.Origin) {
					t.Errorf("origin literal changed: %#v", sent.Origin)
				}
				writeOK(t, w, response)
			}))
			defer closeServer()
			var result map[string]any
			var err error
			if batch {
				result, err = client.BatchAddMessages(context.Background(), "native-v1", []Message{message}, nil)
			} else {
				result, err = client.AddMessage(context.Background(), "native-v1", message.Role, AddMessageOptions{Content: message.Content, Parts: message.Parts, CreatedAt: message.CreatedAt, TurnID: message.TurnID, MessageKind: message.MessageKind, SourceMessageIDs: message.SourceMessageIDs, Origin: message.Origin})
			}
			if err != nil {
				t.Fatal(err)
			}
			if calls != 1 {
				t.Fatalf("calls=%d", calls)
			}
			members, ok := result["accepted_origins"].([]AcceptedOrigin)
			if !ok || len(members) != 1 {
				t.Fatalf("untyped/incomplete receipt: %#v", result)
			}
			statement := members[0].Origin.Statements.Contributors[0].Times[0]
			if statement.Literal != TextTime("2026-10-17T09:10:11.123456+07:00") || statement.Meaning != "source_event" {
				t.Fatalf("time=%#v", statement)
			}
			if members[0].Origin.Statements.Contributors[0].PrimaryAuthor != nil {
				t.Fatal("invented author")
			}
		})
	}
}

func TestOriginSDKRejectsUnconfirmedOrChangedReceipts(t *testing.T) {
	for _, fault := range []string{"omitted", "partial", "literal", "version", "duplicate", "strict", "lost"} {
		t.Run(fault, func(t *testing.T) {
			message, response := originFixture(t)
			members := response["accepted_origins"].([]any)
			origin := members[0].(map[string]any)["origin"].(map[string]any)
			switch fault {
			case "omitted":
				delete(response, "accepted_origins")
			case "partial":
				delete(origin, "receiver")
			case "literal":
				origin["contributors"].([]any)[0].(map[string]any)["times"].([]any)[0].(map[string]any)["literal"] = "2026-10-17T02:10:11.123456Z"
			case "version":
				origin["version"] = 2
			case "duplicate":
				response["accepted_origins"] = append(members, members[0])
			}
			calls := 0
			client, closeServer := testClient(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if fault == "strict" {
					writeAPIError(t, w, 422, "INVALID_ARGUMENT", nil)
					return
				}
				if fault == "lost" {
					w.WriteHeader(200)
					_, _ = w.Write([]byte("{"))
					return
				}
				writeOK(t, w, response)
			}))
			defer closeServer()
			result, err := client.BatchAddMessages(context.Background(), "native-v1", []Message{message}, nil)
			var mismatch *OriginContractMismatch
			if !errors.As(err, &mismatch) || result != nil || calls != 1 {
				t.Fatalf("result=%#v err=%v calls=%d", result, err, calls)
			}
		})
	}
}

func TestOriginTimeScalarAndReceiverBoundary(t *testing.T) {
	for _, literal := range []TimeLiteral{IntegerTime(0), FractionalTime(0.25), TextTime("0")} {
		encoding := "unix_seconds"
		if _, text := literal.(TextTime); text {
			encoding = "opaque"
		}
		value := OriginStatements{Contributors: []ContributorStatement{{Locator: SourceLocator{Provider: "hermes"}, Times: []TimeStatement{{Literal: literal, Encoding: encoding, Meaning: "host_stamp", Basis: FieldBasis{"hermes", "stamp_message_timestamp"}}}}}}
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var restored OriginStatements
		if err := json.Unmarshal(raw, &restored); err != nil {
			t.Fatal(err)
		}
		if restored.Contributors[0].Times[0].Literal != literal {
			t.Fatalf("literal changed: %s", raw)
		}
	}
	message, response := originFixture(t)
	message.Parts = []map[string]any{{"type": "tool"}, {"type": "tool"}}
	if err := validateOriginReadback([]Message{message}, response); err == nil {
		t.Fatal("accepted incomplete fanout")
	}
}
