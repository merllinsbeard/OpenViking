// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0

package openviking

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var originRFC3339 = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:(?:[0-5]\d|60)(?:\.\d+)?(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)$`)

func validateOriginTimestamp(value string) error {
	if !originRFC3339.MatchString(value) {
		return fmt.Errorf("invalid RFC3339 origin literal")
	}
	check := value[:10] + "T" + value[11:]
	if check[17:19] == "60" {
		check = check[:17] + "59" + check[19:]
	}
	if strings.HasSuffix(check, "z") {
		check = strings.TrimSuffix(check, "z") + "Z"
	}
	_, err := time.Parse(time.RFC3339Nano, check)
	return err
}

type FieldBasis struct {
	Producer string `json:"producer"`
	Field    string `json:"field"`
}

type SourceLocator struct {
	Provider    string  `json:"provider"`
	SessionID   *string `json:"session_id,omitempty"`
	SourceScope *string `json:"source_scope,omitempty"`
}

type SourceRef struct {
	Kind  string `json:"kind"`
	Value string `json:"value"`
}

// TimeLiteral preserves a string or numeric observation without converting clocks.
type TimeLiteral interface{ originTimeLiteral() }
type TextTime string
type IntegerTime int64
type FractionalTime float64

func (TextTime) originTimeLiteral()       {}
func (IntegerTime) originTimeLiteral()    {}
func (FractionalTime) originTimeLiteral() {}

type TimeStatement struct {
	Literal  TimeLiteral `json:"literal"`
	Encoding string      `json:"encoding"`
	Meaning  string      `json:"meaning"`
	Basis    FieldBasis  `json:"basis"`
}

func (s *TimeStatement) UnmarshalJSON(data []byte) error {
	var wire struct {
		Literal  json.RawMessage `json:"literal"`
		Encoding string          `json:"encoding"`
		Meaning  string          `json:"meaning"`
		Basis    FieldBasis      `json:"basis"`
	}
	if err := decodeOriginJSON(data, &wire); err != nil {
		return err
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(wire.Literal))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return err
	}
	switch literal := value.(type) {
	case string:
		s.Literal = TextTime(literal)
	case json.Number:
		if integer, err := strconv.ParseInt(string(literal), 10, 64); err == nil {
			s.Literal = IntegerTime(integer)
		} else if number, err := strconv.ParseFloat(string(literal), 64); err == nil && !math.IsInf(number, 0) && !math.IsNaN(number) {
			s.Literal = FractionalTime(number)
		} else {
			return fmt.Errorf("invalid numeric origin time")
		}
	default:
		return fmt.Errorf("origin time must be a string or number")
	}
	s.Encoding, s.Meaning, s.Basis = wire.Encoding, wire.Meaning, wire.Basis
	return s.validate()
}

func (s TimeStatement) validate() error {
	if s.Basis.Producer == "" || s.Basis.Field == "" {
		return fmt.Errorf("origin time requires its field basis")
	}
	switch s.Meaning {
	case "source_event", "provider_record", "host_stamp", "unknown":
	default:
		return fmt.Errorf("invalid origin time meaning")
	}
	switch s.Encoding {
	case "rfc3339":
		value, ok := s.Literal.(TextTime)
		if !ok {
			return fmt.Errorf("RFC3339 requires a text literal")
		}
		return validateOriginTimestamp(string(value))
	case "opaque":
		switch value := s.Literal.(type) {
		case TextTime, IntegerTime:
			return nil
		case FractionalTime:
			if !math.IsInf(float64(value), 0) && !math.IsNaN(float64(value)) {
				return nil
			}
		}
	case "unix_seconds":
		switch value := s.Literal.(type) {
		case IntegerTime:
			return nil
		case FractionalTime:
			if !math.IsInf(float64(value), 0) && !math.IsNaN(float64(value)) {
				return nil
			}
		}
	}
	return fmt.Errorf("invalid declared origin time encoding")
}

type ActorRef struct {
	Identifier string  `json:"identifier"`
	Scope      *string `json:"scope,omitempty"`
}
type ActorClaim struct {
	Actor ActorRef   `json:"actor"`
	Basis FieldBasis `json:"basis"`
}
type ContributorStatement struct {
	Locator         SourceLocator   `json:"locator"`
	Refs            []SourceRef     `json:"refs,omitempty"`
	Times           []TimeStatement `json:"times,omitempty"`
	TransportSender *ActorClaim     `json:"transport_sender,omitempty"`
	PrimaryAuthor   *ActorClaim     `json:"primary_author,omitempty"`
	ContentClass    string          `json:"content_class"`
	ClassBasis      *FieldBasis     `json:"class_basis,omitempty"`
}
type OriginStatements struct {
	Contributors []ContributorStatement `json:"contributors"`
	Version      int                    `json:"version"`
}

func (o OriginStatements) MarshalJSON() ([]byte, error) {
	if o.Version != 0 && o.Version != 1 {
		return nil, fmt.Errorf("unsupported origin version")
	}
	contributors := append([]ContributorStatement{}, o.Contributors...)
	for index := range contributors {
		contributor := &contributors[index]
		if contributor.Locator.Provider == "" {
			return nil, fmt.Errorf("origin requires a provider scope")
		}
		if contributor.ContentClass == "" {
			contributor.ContentClass = "unknown"
		}
		switch contributor.ContentClass {
		case "unknown":
		case "direct_input", "forwarded", "runtime_self_metadata", "bb_relay":
			if contributor.ClassBasis == nil {
				return nil, fmt.Errorf("known content class requires a basis")
			}
		default:
			return nil, fmt.Errorf("invalid origin content class")
		}
		for _, ref := range contributor.Refs {
			if ref.Value == "" {
				return nil, fmt.Errorf("empty origin reference")
			}
			switch ref.Kind {
			case "provider_record", "host_row", "source_message", "supplied_ref":
			default:
				return nil, fmt.Errorf("invalid origin reference kind")
			}
		}
		for _, statement := range contributor.Times {
			if err := statement.validate(); err != nil {
				return nil, err
			}
		}
		for _, claim := range []*ActorClaim{contributor.TransportSender, contributor.PrimaryAuthor} {
			if claim != nil && (claim.Actor.Identifier == "" || claim.Basis.Producer == "" || claim.Basis.Field == "") {
				return nil, fmt.Errorf("actor claim requires an identifier and basis")
			}
		}
		if contributor.ClassBasis != nil && (contributor.ClassBasis.Producer == "" || contributor.ClassBasis.Field == "") {
			return nil, fmt.Errorf("invalid content class basis")
		}
	}
	type wire OriginStatements
	return json.Marshal(wire{Contributors: contributors, Version: 1})
}

type InputApplicability struct {
	Kind        string `json:"kind"`
	GroupID     string `json:"group_id"`
	MemberIndex *int   `json:"member_index,omitempty"`
	MemberCount *int   `json:"member_count,omitempty"`
}

func (a *InputApplicability) UnmarshalJSON(data []byte) error {
	type plain InputApplicability
	var value plain
	if err := decodeOriginJSON(data, &value); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if value.Kind == "input_message" {
		if _, exists := fields["member_index"]; exists {
			return fmt.Errorf("whole-input scope cannot contain member_index")
		}
		if _, exists := fields["member_count"]; exists {
			return fmt.Errorf("whole-input scope cannot contain member_count")
		}
	}
	*a = InputApplicability(value)
	return nil
}

type IngestStamp struct {
	SessionIngestStartedAt string              `json:"session_ingest_started_at"`
	Applicability          *InputApplicability `json:"applicability"`
}
type RecordedOrigin struct {
	Statements OriginStatements
	Receiver   IngestStamp
}
type AcceptedOrigin struct {
	InputIndex       int
	MessageID        string
	Origin           RecordedOrigin
	LegacyCreatedAt  *string
	TurnID           *string
	MessageKind      *string
	SourceMessageIDs []string
}

// OriginContractMismatch means an append may have completed without a matching receipt.
type OriginContractMismatch struct{ Reason string }

func (e *OriginContractMismatch) Error() string { return "openviking origin contract: " + e.Reason }

func decodeOriginJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func validateOriginReadback(messages []Message, result map[string]any) error {
	required := false
	for _, message := range messages {
		required = required || message.Origin != nil
	}
	if !required {
		return nil
	}
	fail := func(reason string) error { return &OriginContractMismatch{Reason: reason} }
	if result["origin_readback"] != "jsonl" {
		return fail("missing persistent readback")
	}
	raw, err := json.Marshal(result["accepted_origins"])
	if err != nil {
		return fail(err.Error())
	}
	var members []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil || len(members) == 0 {
		return fail("missing readback members")
	}
	seenIDs, seenGroups, stamps := map[string]bool{}, map[string]bool{}, map[string]bool{}
	accepted := []AcceptedOrigin{}
	for inputIndex, message := range messages {
		count := 1
		if message.Role == "user" && len(message.Parts) > 1 {
			allTools := true
			for _, part := range message.Parts {
				allTools = allTools && part["type"] == "tool"
			}
			if allTools {
				count = len(message.Parts)
			}
		}
		expected := OriginStatements{}
		if message.Origin != nil {
			expected = *message.Origin
		}
		expectedJSON, err := json.Marshal(expected)
		if err != nil {
			return fail(err.Error())
		}
		found, indices, groups := 0, map[int]bool{}, map[string]bool{}
		for _, fields := range members {
			var index *int
			if err := json.Unmarshal(fields["input_index"], &index); err != nil || index == nil || *index != inputIndex {
				continue
			}
			for _, key := range []string{"input_index", "message_id", "origin", "legacy_created_at", "turn_id", "message_kind", "source_message_ids"} {
				if _, ok := fields[key]; !ok {
					return fail("partial readback member")
				}
			}
			var member struct {
				InputIndex int    `json:"input_index"`
				MessageID  string `json:"message_id"`
				Origin     struct {
					Version      int                     `json:"version"`
					Contributors *[]ContributorStatement `json:"contributors"`
					Receiver     *IngestStamp            `json:"receiver"`
				} `json:"origin"`
				LegacyCreatedAt  *string  `json:"legacy_created_at"`
				TurnID           *string  `json:"turn_id"`
				MessageKind      *string  `json:"message_kind"`
				SourceMessageIDs []string `json:"source_message_ids"`
			}
			memberJSON, _ := json.Marshal(fields)
			if err := decodeOriginJSON(memberJSON, &member); err != nil {
				return fail(err.Error())
			}
			if member.MessageID == "" || seenIDs[member.MessageID] || member.Origin.Version != 1 || member.Origin.Contributors == nil || member.Origin.Receiver == nil {
				return fail("invalid origin member schema")
			}
			seenIDs[member.MessageID] = true
			statements := OriginStatements{Version: 1, Contributors: *member.Origin.Contributors}
			actualJSON, err := json.Marshal(statements)
			if err != nil || !bytes.Equal(actualJSON, expectedJSON) {
				return fail("contributor literals changed")
			}
			receiver := member.Origin.Receiver
			if err := validateOriginTimestamp(receiver.SessionIngestStartedAt); err != nil {
				return fail("invalid receiver timestamp")
			}
			stamps[receiver.SessionIngestStartedAt] = true
			scope := receiver.Applicability
			if scope == nil || scope.GroupID == "" {
				return fail("missing input scope")
			}
			groups[scope.GroupID] = true
			if count == 1 {
				if scope.Kind != "input_message" || scope.MemberIndex != nil || scope.MemberCount != nil {
					return fail("invalid whole-input scope")
				}
			} else {
				if scope.Kind != "input_group" || scope.MemberIndex == nil || scope.MemberCount == nil || *scope.MemberCount != count || *scope.MemberIndex < 0 || *scope.MemberIndex >= count || indices[*scope.MemberIndex] {
					return fail("invalid fanout coverage")
				}
				indices[*scope.MemberIndex] = true
			}
			expectedDate, expectedKind := message.CreatedAt, message.MessageKind
			if expectedDate == "" {
				expectedDate = receiver.SessionIngestStartedAt
			}
			if expectedKind == "" && count > 1 {
				expectedKind = "tool_transport"
			}
			optionalEqual := func(value *string, expected string) bool {
				if expected == "" {
					return value == nil
				}
				return value != nil && *value == expected
			}
			sourceJSON, _ := json.Marshal(member.SourceMessageIDs)
			expectedSourceJSON, _ := json.Marshal(message.SourceMessageIDs)
			if !optionalEqual(member.LegacyCreatedAt, expectedDate) || !optionalEqual(member.TurnID, message.TurnID) || !optionalEqual(member.MessageKind, expectedKind) || !bytes.Equal(sourceJSON, expectedSourceJSON) {
				return fail("source scalar fields changed")
			}
			accepted = append(accepted, AcceptedOrigin{member.InputIndex, member.MessageID, RecordedOrigin{statements, *receiver}, member.LegacyCreatedAt, member.TurnID, member.MessageKind, member.SourceMessageIDs})
			found++
		}
		if found != count || len(groups) != 1 {
			return fail("incomplete input coverage")
		}
		for group := range groups {
			if seenGroups[group] {
				return fail("input group reused")
			}
			seenGroups[group] = true
		}
	}
	if len(accepted) != len(members) || len(stamps) != 1 {
		return fail("unexpected members or ingest clocks")
	}
	result["accepted_origins"] = accepted
	return nil
}

func (m Message) MarshalJSON() ([]byte, error) {
	type plain Message
	wire := struct {
		plain
		SourceIDs *[]string `json:"source_message_ids,omitempty"`
	}{plain: plain(m)}
	if m.SourceMessageIDs != nil {
		wire.SourceIDs = &m.SourceMessageIDs
	}
	return json.Marshal(wire)
}
