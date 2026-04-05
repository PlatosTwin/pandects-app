const fs = require("node:fs");
const path = require("node:path");

const taxonomyResponse = {
  responses: {
    200: {
      description: "OK",
      content: {
        "application/json": {
          schema: {
            type: "object",
            additionalProperties: {
              type: "object",
              required: ["id"],
              properties: {
                id: { type: "string" },
                children: {
                  type: "object",
                  additionalProperties: { type: "object" },
                },
              },
            },
          },
        },
      },
    },
    default: {
      description: "Default error response",
      content: {
        "application/json": {
          schema: {
            type: "object",
            properties: {
              code: { type: "integer", description: "Error code" },
              status: { type: "string", description: "Error name" },
              message: { type: "string", description: "Error message" },
              errors: {
                type: "object",
                description: "Errors",
                additionalProperties: {},
              },
            },
            title: "Error",
          },
        },
      },
    },
  },
};

for (const relativePath of [
  "docs/pandects/get-taxonomy.StatusCodes.json",
  "docs/pandects/get-tax-clause-taxonomy.StatusCodes.json",
]) {
  const filePath = path.join(__dirname, "..", relativePath);
  fs.writeFileSync(filePath, JSON.stringify(taxonomyResponse));
}
