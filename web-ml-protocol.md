# Setup

Input: stdin, text mode, line-buffered

Output: stdout, text mode, line-buffered

# Packet Structure

Single JSON object with a terminating LF.

```JS
{
  "type": "string", // packet type, defines the packet body
  "id": "int", // unique enough packet id
  "data": "various, optional" // packet body
}
```

# Error codes and messages

| Code | Message |
| ---- | ------- |
| `bad-packet-json` | Received packet that is invalid JSON. |
| `null-packet` | Received a null packet. |
| `no-such-packet` | Received packet with invalid type: "${packet}". |
| `no-such-id` | Received a ${packet_type} for invalid packet with id ${id}. |
| `missing-protocol-field` | Received a ${packet_type} without the required protocol field ${field_name}. |
| `missing-body-field` | Received a ${packet_type} without the required body field ${field_name}. |
| `field-out-of-range` | Field ${field_name} is unexpectedly not in range ${range_string}. |
| `field-of-wrong-type` | Field ${field_name} is of invalid type, expected a ${expected_type}. |
| `bug` | Something went wrong on the receiving side. This is a bug. |
| `wrong-pong` | Received a pong with the wrong message. Expected: "${message}", not "${wrong_message}". |
| `path-does-not-exist` | ${path} unexpectedly does not exist. |
| `unexpected-empty-string` | Field ${field_name} is unexpectedly empty. |
| `unexpected-not-directory` | ${path} unexpectedly is not a directory. |
| `unexpected-not-file` | ${path} unexpectedly is not a file. |
| `unexpected-not-file-or-directory` | ${path} unexpectedly is not a file or a directory. |
| `unexpected-class` | "${class}" is not one of previously specified classes. |


# Shared Packets

## `ack`

### Purpose:

Acknowledge receiving a packet if response requires non-trivial processing or there is no meaningful response.

### Body:

```JS
"data": {
  "id": "int" // packet id being acknowledged
}
```

### Expected response:

None


## `error`

### Purpose

To return a failure condition.

### Body:

```JS
"data": {
  "id": "int", // packet id that cause the failure condition
  "code": "string", // error code
  "etc": "string" // unspecified extra information
}
```

### Expected result:

Corresponding `ack`.


## `ping`

### Purpose:

Testing the connection.

### Body:

```JS
"data": {
  "message": "string" // expected response
}
```

### Expected response:

`pong` packet with the same message.


## `pong`

### Purpose:

Testing the connection.

### Body:

```JS
"data": {
  "id": "int", // id of the corresponding ping packet
  "message": "string" // repsonse to a ping
}
```

### Expected response:

Corresponding `ack`.


## `progress`

### Purpose:

Acknowledge progress on a task. Can be sent any amount of times.

### Body:

```JS
"data": {
  "id": "int", // packet id that started the task
  "completeness": "double"
}
```

### Expected response:

Corresponding `ack`.


# Web -> ML packets

## `load-samples`

### Purpose:

Request loading of a folder of samples or a single sample if a file is specified.

### Body:

```JS
"data": {
  "path": "string" // absolute path to the sample folder OR file
  "label": "string" // sample label
}
```

### Expected response:

Corresponding `ack`. Then, any amount of `progress` packets. Then a `progress` packet with `completeness: 1`.


## `infer`

### Purpose:

Request inference on a sample or folder of samples, writing results to a folder or to stdout.

### Body:

```JS
"data": {
  "input_path": "string", // absolute path to the sample folder OR file
  "output_path": "string" // absolute path to the output file OR empty string if output should be written to stdout
}
```

### Behaviour:

If `output_path` is empty, `input_path` MUST be a file. The output is returned as an `inference-result` packet.

If `output_path` is specified and exists, it MUST be a file.

The following JSON output is written to `output_path`:

```JS
{
  "sample-file-path1": "string", // inferred class for the sample in sample-file-path
  "sample-file-path2": "string",
  "sample-file-path3": "string",
  // ...
  "sample-file-pathn": "string",
}
```

### Expected response:

Corresponding `ack`. Then, any amount of `progress` packets. Then a `progress` packet with `completeness: 1`.


# ML -> Web packets

## `inference-result`

### Purpose:

Return a single inference result.

### Body:

```JS
"data": {
  "label": "string"
}
```

### Expected response:

Corresponding `ack`.
