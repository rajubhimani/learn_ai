import asyncio
import re
from typing import Any, Dict, List
import boto3
from botocore.exceptions import ClientError
from mcp.server.fastmcp import FastMCP
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword

class FullyReadOnlyAthenaMCPServer:
    def __init__(self, region: str = 'us-east-1', workgroup: str = 'primary', 
                 default_catalog: str = None):
        self.athena_client = boto3.client('athena', region_name=region)
        self.glue_client = boto3.client('glue', region_name=region)
        self.workgroup = workgroup
        self.default_catalog = default_catalog
        self.available_catalogs = {}
        
        # Comprehensive read-only validation
        self.allowed_keywords = {
            'SELECT', 'DESCRIBE', 'DESC', 'SHOW', 'EXPLAIN', 'VALUES', 'WITH'
        }
        
        self.forbidden_keywords = {
            # DML Operations
            'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'UPSERT', 'REPLACE',
            # DDL Operations
            'CREATE', 'DROP', 'ALTER', 'TRUNCATE', 'RENAME',
            # Transaction Control
            'COMMIT', 'ROLLBACK', 'SAVEPOINT',
            # Stored Procedures/Functions
            'CALL', 'EXEC', 'EXECUTE',
            # Data Control
            'GRANT', 'REVOKE', 'DENY',
            # System Commands
            'SET', 'RESET', 'USE',
            # Bulk Operations
            'COPY', 'BULK', 'LOAD', 'IMPORT', 'EXPORT',
            # Administrative
            'VACUUM', 'ANALYZE', 'REINDEX', 'CLUSTER'
        }
        
        # Dangerous functions that could modify data or system state
        self.forbidden_functions = {
            'msck', 'refresh', 'invalidate_metadata', 'repair',
            'load_data', 'insert_overwrite', 'create_table_as'
        }
        
        # Initialize FastMCP
        self.mcp = FastMCP("athena-fully-readonly")
        self._register_tools()
        
        # Discover available catalogs on startup
        asyncio.create_task(self._discover_catalogs())
    
    async def _discover_catalogs(self):
        """Discover all available catalogs in the account"""
        try:
            response = self.athena_client.list_data_catalogs()
            for catalog in response.get('DataCatalogsSummary', []):
                self.available_catalogs[catalog['CatalogName']] = {
                    'type': catalog['Type'],
                    'connection_type': catalog.get('ConnectionType'),
                    'status': catalog['Status']
                }
        except Exception as e:
            print(f"Warning: Could not discover catalogs: {e}")
    
    def _register_tools(self):
        """Register all MCP tools with enhanced read-only validation"""
        
        @self.mcp.tool()
        async def list_catalogs() -> Dict[str, Any]:
            """List all available data catalogs in the account (READ-ONLY)"""
            try:
                response = self.athena_client.list_data_catalogs()
                catalogs = []
                for catalog in response.get('DataCatalogsSummary', []):
                    catalogs.append({
                        'name': catalog['CatalogName'],
                        'type': catalog['Type'],
                        'connection_type': catalog.get('ConnectionType'),
                        'status': catalog['Status']
                    })
                return {"catalogs": catalogs, "operation": "READ-ONLY"}
            except ClientError as e:
                return {"error": f"Failed to list catalogs: {str(e)}"}
        
        @self.mcp.tool()
        async def execute_query(query: str, catalog: str = None, workgroup: str = None) -> Dict[str, Any]:
            """Execute a STRICTLY READ-ONLY SQL query in Athena against any catalog"""
            
            # Multi-layer security validation
            validation_result = self._comprehensive_query_validation(query)
            if not validation_result['is_valid']:
                return {
                    "error": f"SECURITY VIOLATION: {validation_result['reason']}",
                    "query_rejected": True,
                    "security_policy": "STRICT_READ_ONLY"
                }
            
            try:
                target_catalog = catalog or self.default_catalog
                target_workgroup = workgroup or self.workgroup
                
                # Add security metadata to query execution
                execution_params = {
                    'QueryString': query,
                    'WorkGroup': target_workgroup
                }
                
                if target_catalog:
                    execution_params['QueryExecutionContext'] = {
                        'Catalog': target_catalog
                    }
                
                # Start query execution
                response = self.athena_client.start_query_execution(**execution_params)
                query_execution_id = response['QueryExecutionId']
                
                # Wait for completion
                result = await self._wait_for_query_completion(query_execution_id)
                
                if result['status'] == 'SUCCEEDED':
                    results = await self._get_query_results(query_execution_id)
                    return {
                        "query_execution_id": query_execution_id,
                        "status": "success",
                        "catalog": target_catalog,
                        "results": results,
                        "security_policy": "READ_ONLY_VALIDATED",
                        "query_type": validation_result['query_type']
                    }
                else:
                    return {
                        "query_execution_id": query_execution_id,
                        "status": "failed",
                        "catalog": target_catalog,
                        "error": result.get('error', 'Query execution failed'),
                        "security_policy": "READ_ONLY_VALIDATED"
                    }
                    
            except ClientError as e:
                return {"error": f"AWS Error: {str(e)}", "security_policy": "READ_ONLY_VALIDATED"}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}", "security_policy": "READ_ONLY_VALIDATED"}
        
        @self.mcp.tool()
        async def list_databases(catalog: str = None) -> Dict[str, Any]:
            """List databases in any catalog (READ-ONLY METADATA OPERATION)"""
            target_catalog = catalog or self.default_catalog or 'AwsDataCatalog'
            
            try:
                catalog_info = self.available_catalogs.get(target_catalog, {})
                catalog_type = catalog_info.get('type', 'GLUE')
                
                if catalog_type == 'GLUE' and target_catalog == 'AwsDataCatalog':
                    # Use Glue API for AWS Glue Data Catalog (inherently read-only)
                    response = self.glue_client.get_databases()
                    databases = [db['Name'] for db in response['DatabaseList']]
                    return {
                        "catalog": target_catalog, 
                        "databases": databases,
                        "operation": "READ_ONLY_METADATA"
                    }
                else:
                    # Use validated SQL query for federated catalogs
                    query = f"SHOW DATABASES"
                    if target_catalog:
                        query = f"SHOW DATABASES IN {target_catalog}"
                    
                    # Validate the SHOW command (extra security layer)
                    if not self._is_metadata_query(query):
                        return {"error": "Invalid metadata query", "security_violation": True}
                    
                    result = await self._execute_validated_sql_query(query, target_catalog)
                    if 'error' in result:
                        return result
                    
                    databases = [row.get('database_name', row.get('Database', '')) 
                               for row in result.get('results', [])]
                    return {
                        "catalog": target_catalog, 
                        "databases": databases,
                        "operation": "READ_ONLY_METADATA"
                    }
                    
            except ClientError as e:
                return {"error": f"Failed to list databases: {str(e)}"}
        
        @self.mcp.tool()
        async def list_tables(database: str, catalog: str = None) -> Dict[str, Any]:
            """List tables in a database (READ-ONLY METADATA OPERATION)"""
            target_catalog = catalog or self.default_catalog or 'AwsDataCatalog'
            
            try:
                catalog_info = self.available_catalogs.get(target_catalog, {})
                catalog_type = catalog_info.get('type', 'GLUE')
                
                if catalog_type == 'GLUE' and target_catalog == 'AwsDataCatalog':
                    # Use Glue API (inherently read-only)
                    response = self.glue_client.get_tables(DatabaseName=database)
                    tables = [table['Name'] for table in response['TableList']]
                    return {
                        "catalog": target_catalog, 
                        "database": database, 
                        "tables": tables,
                        "operation": "READ_ONLY_METADATA"
                    }
                else:
                    # Use validated SQL query
                    query = f"SHOW TABLES IN {target_catalog}.{database}"
                    
                    if not self._is_metadata_query(query):
                        return {"error": "Invalid metadata query", "security_violation": True}
                    
                    result = await self._execute_validated_sql_query(query, target_catalog)
                    if 'error' in result:
                        return result
                    
                    tables = [row.get('tab_name', row.get('Table', '')) 
                             for row in result.get('results', [])]
                    return {
                        "catalog": target_catalog, 
                        "database": database, 
                        "tables": tables,
                        "operation": "READ_ONLY_METADATA"
                    }
                    
            except ClientError as e:
                return {"error": f"Failed to list tables: {str(e)}"}
        
        @self.mcp.tool()
        async def describe_table(database: str, table: str, catalog: str = None) -> Dict[str, Any]:
            """Get table schema (READ-ONLY METADATA OPERATION)"""
            target_catalog = catalog or self.default_catalog or 'AwsDataCatalog'
            
            try:
                catalog_info = self.available_catalogs.get(target_catalog, {})
                catalog_type = catalog_info.get('type', 'GLUE')
                
                if catalog_type == 'GLUE' and target_catalog == 'AwsDataCatalog':
                    # Use Glue API (inherently read-only)
                    response = self.glue_client.get_table(DatabaseName=database, Name=table)
                    table_info = response['Table']
                    columns = [
                        {"name": col['Name'], "type": col['Type']} 
                        for col in table_info['StorageDescriptor']['Columns']
                    ]
                    return {
                        "catalog": target_catalog,
                        "database": database,
                        "table": table,
                        "columns": columns,
                        "location": table_info['StorageDescriptor'].get('Location'),
                        "operation": "READ_ONLY_METADATA"
                    }
                else:
                    # Use validated SQL query
                    query = f"DESCRIBE {target_catalog}.{database}.{table}"
                    
                    if not self._is_metadata_query(query):
                        return {"error": "Invalid metadata query", "security_violation": True}
                    
                    result = await self._execute_validated_sql_query(query, target_catalog)
                    if 'error' in result:
                        return result
                    
                    return {
                        "catalog": target_catalog,
                        "database": database,
                        "table": table,
                        "schema": result.get('results', []),
                        "operation": "READ_ONLY_METADATA"
                    }
                    
            except ClientError as e:
                return {"error": f"Failed to describe table: {str(e)}"}
        
        @self.mcp.tool()
        async def list_workgroups() -> Dict[str, Any]:
            """List all available Athena workgroups (READ-ONLY SYSTEM INFO)"""
            try:
                response = self.athena_client.list_work_groups()
                workgroups = []
                for wg in response['WorkGroups']:
                    workgroups.append({
                        'name': wg['Name'],
                        'state': wg['State'],
                        'description': wg.get('Description', '')
                    })
                return {
                    "workgroups": workgroups,
                    "operation": "READ_ONLY_SYSTEM_INFO"
                }
            except ClientError as e:
                return {"error": f"Failed to list workgroups: {str(e)}"}
        
        @self.mcp.tool()
        async def get_catalog_info(catalog_name: str) -> Dict[str, Any]:
            """Get detailed catalog information (READ-ONLY SYSTEM INFO)"""
            try:
                response = self.athena_client.get_data_catalog(Name=catalog_name)
                catalog = response['DataCatalog']
                return {
                    "name": catalog['Name'],
                    "type": catalog['Type'],
                    "description": catalog.get('Description', ''),
                    "parameters": catalog.get('Parameters', {}),
                    "status": catalog.get('Status', 'Unknown'),
                    "operation": "READ_ONLY_SYSTEM_INFO"
                }
            except ClientError as e:
                return {"error": f"Failed to get catalog info: {str(e)}"}
    
    def _comprehensive_query_validation(self, query: str) -> Dict[str, Any]:
        """Multi-layer security validation for queries"""
        
        # Layer 1: Basic sanitization
        query_clean = query.strip().upper()
        
        # Layer 2: Check for forbidden keywords
        for forbidden in self.forbidden_keywords:
            if re.search(r'\b' + forbidden + r'\b', query_clean):
                return {
                    'is_valid': False,
                    'reason': f'Forbidden keyword detected: {forbidden}',
                    'security_level': 'CRITICAL'
                }
        
        # Layer 3: Check for dangerous functions
        for func in self.forbidden_functions:
            if re.search(r'\b' + func + r'\b', query_clean):
                return {
                    'is_valid': False,
                    'reason': f'Dangerous function detected: {func}',
                    'security_level': 'HIGH'
                }
        
        # Layer 4: Parse SQL and validate structure
        try:
            parsed = sqlparse.parse(query.strip())
            if not parsed:
                return {
                    'is_valid': False,
                    'reason': 'Invalid SQL syntax',
                    'security_level': 'MEDIUM'
                }
            
            query_types = []
            for statement in parsed:
                validation = self._validate_statement_structure(statement)
                if not validation['is_valid']:
                    return validation
                query_types.append(validation['type'])
            
            return {
                'is_valid': True,
                'query_type': query_types,
                'security_level': 'VALIDATED'
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'reason': f'SQL parsing error: {str(e)}',
                'security_level': 'HIGH'
            }
    
    def _validate_statement_structure(self, statement: Statement) -> Dict[str, Any]:
        """Validate individual SQL statement structure"""
        first_keyword = None
        
        for token in statement.flatten():
            if token.ttype is Keyword:
                keyword = token.value.upper()
                if not first_keyword:
                    first_keyword = keyword
                
                # Check if this is an allowed keyword
                if keyword in self.allowed_keywords:
                    continue
                elif keyword in self.forbidden_keywords:
                    return {
                        'is_valid': False,
                        'reason': f'Forbidden SQL operation: {keyword}',
                        'security_level': 'CRITICAL'
                    }
        
        if first_keyword not in self.allowed_keywords:
            return {
                'is_valid': False,
                'reason': f'Query must start with allowed keyword, found: {first_keyword}',
                'security_level': 'HIGH'
            }
        
        return {
            'is_valid': True,
            'type': first_keyword,
            'security_level': 'VALIDATED'
        }
    
    def _is_metadata_query(self, query: str) -> bool:
        """Validate that query is a metadata-only operation"""
        query_upper = query.strip().upper()
        metadata_patterns = [
            r'^SHOW\s+(DATABASES|TABLES|COLUMNS)',
            r'^DESCRIBE\s+',
            r'^DESC\s+',
            r'^EXPLAIN\s+'
        ]
        
        return any(re.match(pattern, query_upper) for pattern in metadata_patterns)
    
    async def _execute_validated_sql_query(self, query: str, catalog: str = None) -> Dict[str, Any]:
        """Execute SQL query with additional validation layer"""
        
        # Double-check validation before execution
        validation = self._comprehensive_query_validation(query)
        if not validation['is_valid']:
            return {"error": f"Security validation failed: {validation['reason']}"}
        
        try:
            execution_params = {
                'QueryString': query,
                'WorkGroup': self.workgroup
            }
            
            if catalog:
                execution_params['QueryExecutionContext'] = {'Catalog': catalog}
            
            response = self.athena_client.start_query_execution(**execution_params)
            query_execution_id = response['QueryExecutionId']
            
            result = await self._wait_for_query_completion(query_execution_id)
            
            if result['status'] == 'SUCCEEDED':
                results = await self._get_query_results(query_execution_id)
                return {"results": results, "security_validated": True}
            else:
                return {"error": result.get('error', 'Query failed'), "security_validated": True}
                
        except Exception as e:
            return {"error": str(e), "security_validated": True}
    
    async def _wait_for_query_completion(self, query_execution_id: str, max_retries: int = 100) -> Dict[str, Any]:
        """Wait for query execution to complete"""
        for _ in range(max_retries):
            try:
                response = self.athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                status = response['QueryExecution']['Status']['State']
                
                if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    result = {"status": status}
                    if status == 'FAILED':
                        result["error"] = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                    return result
                
                await asyncio.sleep(0.5)
                
            except ClientError as e:
                return {"status": "ERROR", "error": str(e)}
        
        return {"status": "TIMEOUT", "error": "Query execution timed out"}
    
    async def _get_query_results(self, query_execution_id: str, max_results: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve results from a completed query"""
        try:
            response = self.athena_client.get_query_results(
                QueryExecutionId=query_execution_id,
                MaxResults=max_results
            )
            
            result_set = response['ResultSet']
            
            if not result_set['Rows']:
                return []
            
            # Extract column names from the first row (header)
            columns = [col['VarCharValue'] for col in result_set['Rows'][0]['Data']]
            
            # Extract data rows (skip header row)
            rows = []
            for row in result_set['Rows'][1:]:
                row_data = {}
                for i, data in enumerate(row['Data']):
                    row_data[columns[i]] = data.get('VarCharValue', '')
                rows.append(row_data)
            
            return rows
            
        except ClientError as e:
            raise Exception(f"Failed to get query results: {str(e)}")

# Server startup with enhanced security logging
if __name__ == "__main__":
    import sys
    import os
    import logging
    
    # Enhanced security logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - ATHENA_READONLY_MCP - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Fully Read-Only Athena MCP Server with Enhanced Security")
    
    # Configuration from environment variables
    region = os.getenv('AWS_REGION', 'us-east-1')
    workgroup = os.getenv('ATHENA_WORKGROUP', 'primary')
    default_catalog = os.getenv('DEFAULT_CATALOG')
    
    # Initialize the fully read-only server
    server = FullyReadOnlyAthenaMCPServer(
        region=region, 
        workgroup=workgroup, 
        default_catalog=default_catalog
    )
    
    logger.info(f"Server initialized with workgroup: {workgroup}, region: {region}")
    logger.info("SECURITY POLICY: STRICT READ-ONLY MODE ENABLED")
    
    # Run the MCP server
    server.mcp.run()
