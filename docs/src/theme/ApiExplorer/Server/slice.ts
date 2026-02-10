import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import type { ServerObject } from "docusaurus-plugin-openapi-docs/src/openapi/types";

export interface State {
  value?: ServerObject;
  options: ServerObject[];
}

const initialState: State = {} as State;

export const slice = createSlice({
  name: "server",
  initialState,
  reducers: {
    setServer: (state, action: PayloadAction<string>) => {
      const selectedServer = JSON.parse(action.payload) as ServerObject;
      state.value =
        state.options.find((server) => server.url === selectedServer.url) ??
        selectedServer;
    },
    setServerVariable: (state, action: PayloadAction<string>) => {
      if (state.value?.variables) {
        const parsedPayload = JSON.parse(action.payload);
        state.value.variables[parsedPayload.key].default = parsedPayload.value;
      }
    },
  },
});

export const { setServer, setServerVariable } = slice.actions;

export default slice.reducer;
