// Agreement Types

export interface Agreement {
<<<<<<< HEAD
=======
  id: string;
>>>>>>> b67e1464ae3dac5ea6912901280ebad2df92dbdd
  year: string;
  target: string;
  acquirer: string;
  xml: string;
<<<<<<< HEAD
=======
  agreementUuid: string;
}

export interface AgreementResponse {
  agreement: Agreement;
>>>>>>> b67e1464ae3dac5ea6912901280ebad2df92dbdd
}

// Table of Contents types for XML navigation
export interface TOCItem {
  id: string;
  title: string;
  level: number;
  sectionUuid?: string;
  children?: TOCItem[];
}
