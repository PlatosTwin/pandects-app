// Agreement Types

export interface Agreement {
  id: string;
  year: string;
  target: string;
  acquirer: string;
  xml: string;
  agreementUuid: string;
}

export interface AgreementResponse {
  agreement: Agreement;
}

// Table of Contents types for XML navigation
export interface TOCItem {
  id: string;
  title: string;
  level: number;
  sectionUuid?: string;
  children?: TOCItem[];
}
